import os
import io
import sys
import math
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import streamlit as st

# =========================
# 1) U-Net
# =========================

class UNet(nn.Module):
    def conv_plus_conv(sefl, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def __init__(self, channels_in=3, channels_out=1):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.base_channels = 32

        self.enc_conv0 = self.conv_plus_conv(3, self.base_channels)
        self.pool0 = nn.MaxPool2d(2)
        self.enc_conv1 = self.conv_plus_conv(self.base_channels, self.base_channels*2)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = self.conv_plus_conv(self.base_channels*2, self.base_channels*4)
        self.pool2 = nn.MaxPool2d(2)
        self.enc_conv3 = self.conv_plus_conv(self.base_channels*4, self.base_channels*8)
        self.pool3 = nn.MaxPool2d(2)
        self.enc_conv4 = self.conv_plus_conv(self.base_channels*8, self.base_channels*16)
        self.pool4 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck_conv = self.conv_plus_conv(self.base_channels*16, self.base_channels*16)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
        self.dec_conv0 = self.conv_plus_conv(self.base_channels*2, self.base_channels)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
        self.dec_conv1 = self.conv_plus_conv(self.base_channels*4, self.base_channels)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
        self.dec_conv2 = self.conv_plus_conv(self.base_channels*8, self.base_channels*2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
        self.dec_conv3 = self.conv_plus_conv(self.base_channels*16, self.base_channels*4)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  #
        self.dec_conv4 = self.conv_plus_conv(self.base_channels*32, self.base_channels*8)

        self.out = nn.Conv2d(in_channels=self.base_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        e0 = self.enc_conv0(x)
        x = self.pool0(e0)
        e1 = self.enc_conv1(x)
        x = self.pool1(e1)
        e2 = self.enc_conv2(x)
        x = self.pool2(e2)
        e3 = self.enc_conv3(x)
        x = self.pool3(e3)
        e4 = self.enc_conv4(x)
        x = self.pool4(e4)

        # bottleneck
        x = self.bottleneck_conv(x)

        # decoder
        d0 = self.upsample4(x)
        x = torch.cat((d0, e4), dim=1)
        x = self.dec_conv4(x)

        d1 = self.upsample3(x)
        x = torch.cat((d1, e3), dim=1)
        x = self.dec_conv3(x)

        d2 = self.upsample2(x)
        x = torch.cat((d2, e2), dim=1)
        x = self.dec_conv2(x)

        d3 = self.upsample1(x)
        x = torch.cat((d3, e1), dim=1)
        x = self.dec_conv1(x)

        d4 = self.upsample0(x)
        x = torch.cat((d4, e0), dim=1)
        x = self.dec_conv0(x)

        out = self.out(x)
        return out

# =========================
# 2) Настройки страницы
# =========================
st.set_page_config(
    page_title="U-Net Segmentation Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("U-Net Сегментация изображения")
st.write("Загрузите изображение, выберите чекпойнт и получите предсказанную маску.")

# =========================
# 3) Вспомогательные функции
# =========================
@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device, n_channels: int = 3, n_classes: int = 1):
    model = UNet(channels_in=n_channels, channels_out=n_classes)
    # Если чекпойнт сохранен с map_location специфичным, используем безопасную загрузку:
    state = torch.load(checkpoint_path, map_location=device)
    # Поддержка случая, если сохранён dict {'model_state_dict': ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Возможно сохранение с префиксом "module."
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "")
        new_state[new_k] = v
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model

def preprocess_image(pil_img: Image.Image, input_size: Tuple[int, int], normalize: str = "imagenet", to_float32: bool = True):
    # Конвертация в RGB
    img = pil_img.convert("RGB")
    # Ресайз к целевому размеру 
    img_resized = img.resize(input_size[::-1], Image.BILINEAR)  # input_size = (H, W)
    img_np = np.array(img_resized).astype(np.float32 if to_float32 else np.uint8)

    if to_float32:
        img_np /= 255.0

    if normalize == "none":
        pass
    elif normalize == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
    else:
        pass

    # HWC -> CHW
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)  # [1,3,H,W]
    return tensor, img_resized  # вернем также пилку для визуализации

def postprocess_mask(logits: torch.Tensor, orig_size: Tuple[int, int], threshold: float = 0.5):
    # logits: [1,1,H,W]
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float()  # [1,1,H,W]
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255  # [H,W] 0/255
    # Возврат к оригинальному размеру
    mask_img = Image.fromarray(mask_np)
    mask_img = mask_img.resize(orig_size[::-1], Image.NEAREST)
    return mask_img

def overlay_mask(image_pil: Image.Image, mask_pil: Image.Image, color=(255, 0, 0), alpha=0.4):
    img = image_pil.convert("RGBA")
    mask = mask_pil.convert("L")
    # создаем цветную маску
    color_img = Image.new("RGBA", img.size, color + (0,))
    # используем маску как альфа-канал для заливки
    color_alpha = Image.new("L", img.size, int(255 * alpha))
    masked_color = Image.merge("RGBA", (
        Image.new("L", img.size, color[0]),
        Image.new("L", img.size, color[1]),
        Image.new("L", img.size, color[2]),
        Image.composite(color_alpha, Image.new("L", img.size, 0), mask)
    ))
    blended = Image.alpha_composite(img, masked_color)
    return blended.convert("RGB")

def to_device(t: torch.Tensor, device: torch.device):
    return t.to(device, non_blocking=True)

# =========================
# 4) Боковая панель (настройки)
# =========================
st.sidebar.header("Настройки")

# Папка с моделями
default_model_dir = "."
model_dir = st.sidebar.text_input("Папка с чекпойнтами", value=default_model_dir)

# Сканируем доступные .pth
available_ckpts = []
if os.path.isdir(model_dir):
    for f in os.listdir(model_dir):
        if f.endswith(".pth"):
            available_ckpts.append(f)
available_ckpts = sorted(available_ckpts)

if not available_ckpts:
    st.sidebar.warning("В папке нет .pth файлов. Поместите best_model.pth и/или другие чекпойнты в выбранную папку.")

selected_ckpt = st.sidebar.selectbox(
    "Выберите чекпойнт модели",
    options=available_ckpts if available_ckpts else ["best_BCE_model.pth", "best_dice_model.pth"],
    index=0
)

input_h = st.sidebar.number_input("Высота входа (H)", min_value=64, max_value=2048, value=256, step=32)
input_w = st.sidebar.number_input("Ширина входа (W)", min_value=64, max_value=2048, value=256, step=32)
threshold = st.sidebar.slider("Порог бинаризации", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
show_overlay = st.sidebar.checkbox("Показывать оверлей", value=True)
overlay_alpha = st.sidebar.slider("Прозрачность оверлея", 0.0, 1.0, 0.4, 0.05)
normalize_choice = st.sidebar.selectbox("Нормализация", ["none", "imagenet"], index=0)
device_choice = st.sidebar.selectbox("Устройство", ["auto", "cpu", "cuda"], index=0)

# =========================
# 5) Определяем устройство
# =========================
if device_choice == "cuda":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif device_choice == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.write(f"Используется устройство: {device}")

# =========================
# 6) Загрузка модели
# =========================
ckpt_path = os.path.join(model_dir, selected_ckpt)
model = None
model_load_error = None
if os.path.exists(ckpt_path):
    try:
        with st.spinner(f"Загрузка модели из {ckpt_path}..."):
            model = load_model(ckpt_path, device=device, n_channels=3, n_classes=1)
        st.sidebar.success("Модель загружена.")
    except Exception as e:
        model_load_error = str(e)
        st.sidebar.error(f"Ошибка загрузки модели: {e}")
else:
    st.sidebar.info(f"Ожидается файл: {ckpt_path}")

# =========================
# 7) Загрузка изображения
# =========================
uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg", "bmp", "tiff"])

# =========================
# 8) Инференс
# =========================
col_left, col_right = st.columns(2)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Не удалось открыть изображение: {e}")
        st.stop()

    orig_size = image.size[::-1]  # (H,W)
    with col_left:
        st.subheader("Оригинальное изображение")
        st.image(image, use_container_width=True)

    if model is None:
        st.warning("Модель не загружена. Выберите корректный чекпойнт.")
        st.stop()

    run = st.button("Запустить сегментацию", type="primary")

    if run:
        with st.spinner("Выполняется инференс..."):
            try:
                tensor, resized_for_vis = preprocess_image(
                    image, input_size=(input_h, input_w), normalize=normalize_choice
                )
                tensor = to_device(tensor, device)
                with torch.no_grad():
                    logits = model(tensor)  # [1,1,H,W]
                # Маску вернем к размеру исходного изображения
                mask_img = postprocess_mask(logits, orig_size=orig_size, threshold=threshold)

                with col_right:
                    st.subheader("Результат сегментации")
                    if show_overlay:
                        overlay = overlay_mask(image, mask_img, color=(255, 0, 0), alpha=overlay_alpha)
                        st.image(overlay, caption="Оверлей маски", use_container_width=True)
                    st.image(mask_img, caption="Предсказанная маска", use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка при инференсе: {e}")

else:
    st.info("Загрузите изображение для начала.")

# =========================
# 9) Пример пакетной визуализации (опционально)
# =========================
with st.expander("Пакетная обработка (опционально)"):
    st.write("Можно перетащить несколько изображений и обработать все сразу.")
    files = st.file_uploader("Загрузите несколько изображений", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files and model is not None:
        batch_btn = st.button("Обработать пакет")
        if batch_btn:
            with st.spinner("Обработка..."):
                results = []
                for f in files:
                    try:
                        img = Image.open(f).convert("RGB")
                        orig_size = img.size[::-1]
                        tensor, _ = preprocess_image(img, input_size=(input_h, input_w), normalize=normalize_choice)
                        tensor = to_device(tensor, device)
                        with torch.no_grad():
                            logits = model(tensor)
                        mask_img = postprocess_mask(logits, orig_size=orig_size, threshold=threshold)
                        if show_overlay:
                            overlay = overlay_mask(img, mask_img, color=(255, 0, 0), alpha=overlay_alpha)
                            results.append((img, overlay, mask_img))
                        else:
                            results.append((img, None, mask_img))
                    except Exception as e:
                        st.warning(f"{f.name}: ошибка — {e}")

                for i, (img, overlay, mask) in enumerate(results):
                    st.write(f"Изображение {i+1}")
                    cols = st.columns(3)
                    cols[0].image(img, caption="Оригинал", use_container_width=True)
                    if overlay is not None:
                        cols[1].image(overlay, caption="Оверлей", use_container_width=True)
                    cols[2].image(mask, caption="Маска", use_container_width=True)
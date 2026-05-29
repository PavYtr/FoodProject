from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def ensure_pil_image(image: Any) -> Image.Image:
    if image is None:
        raise ValueError("Image is required")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            return Image.fromarray(array.astype("uint8"), mode="L").convert("RGB")
        if array.shape[-1] == 4:
            return Image.fromarray(array.astype("uint8"), mode="RGBA").convert("RGB")
        return Image.fromarray(array.astype("uint8")).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def quality_warnings(image: Image.Image) -> list[str]:
    warnings: list[str] = []
    width, height = image.size
    if min(width, height) < 256:
        warnings.append("Изображение маленькое: сегментация и оценка массы могут быть неточными.")

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    mean_light = float(gray.mean())
    if mean_light < 0.15:
        warnings.append("Изображение темное: качество классификации может снизиться.")
    if mean_light > 0.92:
        warnings.append("Изображение очень светлое: маски могут быть нестабильными.")

    return warnings


def foreground_mask(image: Image.Image) -> np.ndarray:
    rgb = to_numpy_rgb(image).astype(np.float32) / 255.0
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = max_channel - min_channel

    mask = (saturation > 0.10) & (max_channel > 0.18)
    if mask.mean() < 0.02:
        luminance = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        mask = (luminance > 0.08) & (luminance < 0.92)

    mask_img = Image.fromarray((mask.astype("uint8") * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.MedianFilter(size=5))
    return np.asarray(mask_img) > 127

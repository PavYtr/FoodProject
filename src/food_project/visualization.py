from __future__ import annotations

import numpy as np
from PIL import Image

from food_project.schemas import SegmentPrediction


PALETTE = [
    (220, 74, 74),
    (57, 145, 230),
    (84, 177, 102),
    (235, 169, 62),
    (153, 107, 220),
]


def overlay_segments(
    image: Image.Image,
    segments: list[SegmentPrediction],
    alpha: float = 0.42,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()

    for index, segment in enumerate(segments):
        if segment.mask is None:
            continue
        mask = np.asarray(segment.mask, dtype=bool)
        color = np.asarray(PALETTE[index % len(PALETTE)], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    return Image.fromarray(np.clip(overlay, 0, 255).astype("uint8"))


def depth_to_image(depth_map: np.ndarray | None) -> Image.Image | None:
    if depth_map is None:
        return None
    values = np.asarray(depth_map, dtype=np.float32)
    if values.ndim == 3:
        values = values[:, :, 0]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        array = np.zeros(values.shape[:2], dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [5, 95])
        scale = max(float(high - low), 1e-6)
        array = (np.clip((values - low) / scale, 0.0, 1.0) * 255).astype(np.uint8)
    if array.ndim == 3:
        array = array[:, :, 0]
    rgb = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = array
    rgb[:, :, 1] = np.clip(255 - np.abs(array.astype(np.int16) - 128) * 2, 0, 255)
    rgb[:, :, 2] = 255 - array
    return Image.fromarray(rgb, mode="RGB")

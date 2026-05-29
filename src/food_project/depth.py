from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class DepthEstimator:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("DEPTH_MODEL", "depth-anything-v3-base")
        self.status = "not_loaded"
        self.pipe: Any | None = None
        self.device = "cpu"
        self.load_error: str | None = None

    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float], list[str]]:
        depth = self._predict_model_depth(image)
        if depth is not None:
            return depth, _depth_stats(depth), []

        depth = _heuristic_depth(image)
        suffix = f" ({self.load_error})" if self.load_error else ""
        warnings = [
            "Depth map рассчитан эвристически; "
            f"depth-estimation недоступен{suffix}."
        ]
        self.status = "heuristic"
        return depth, _depth_stats(depth), warnings

    def _predict_model_depth(self, image: Image.Image) -> np.ndarray | None:
        if str(self.model_id).strip().lower() in {"", "none", "disabled", "heuristic"}:
            self.load_error = "disabled by config"
            return None

        if self.pipe is None and self.load_error is None:
            try:
                import torch
                from depth_estimation import pipeline

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipe = pipeline(
                    "depth-estimation",
                    model=self.model_id,
                    device=self.device,
                )
                self.status = "loaded"
            except Exception as exc:  # pragma: no cover - optional external model
                self.load_error = str(exc)
                return None

        if self.pipe is None:
            return None

        try:
            import torch

            with torch.no_grad():
                result = self.pipe(image.convert("RGB"))
            depth = getattr(result, "depth", result)
            self.status = "loaded"
            return np.asarray(depth, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - optional external model
            self.load_error = str(exc)
            self.status = "heuristic"
            return None


def _heuristic_depth(image: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(radius=5))
    array = np.asarray(gray, dtype=np.float32) / 255.0

    height, width = array.shape
    y, x = np.ogrid[:height, :width]
    center_prior = 1.0 - np.sqrt(
        ((x - width / 2.0) / max(width / 2.0, 1.0)) ** 2
        + ((y - height / 2.0) / max(height / 2.0, 1.0)) ** 2
    )
    center_prior = np.clip(center_prior, 0.0, 1.0)

    depth = 0.55 * (1.0 - array) + 0.45 * center_prior
    finite = depth[np.isfinite(depth)]
    if finite.size:
        min_depth = float(finite.min())
        max_depth = float(finite.max())
        if max_depth > min_depth:
            depth = (depth - min_depth) / (max_depth - min_depth)
    return depth.astype(np.float32)


def _normalize_for_stats(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim == 3:
        values = values[:, :, 0]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape[:2], dtype=np.float32)

    low, high = np.percentile(finite, [5, 95])
    scale = max(float(high - low), 1e-6)
    return np.clip((values - low) / scale, 0.0, 1.0).astype(np.float32)


def _depth_stats(depth: np.ndarray) -> dict[str, float]:
    normalized = _normalize_for_stats(depth)
    return {
        "depth_mean": float(normalized.mean()),
        "depth_std": float(normalized.std()),
        "depth_min": float(normalized.min()),
        "depth_max": float(normalized.max()),
    }

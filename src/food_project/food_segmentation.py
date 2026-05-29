from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from food_project.preprocessing import foreground_mask
from food_project.schemas import SegmentPrediction


class YOLOSegmentationModel:
    def __init__(
        self,
        name: str,
        model_path: Path,
        device: str = "cpu",
        image_size: int = 640,
        confidence: float = 0.25,
        fallback_kind: str = "food",
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.fallback_kind = fallback_kind
        self.model: Any | None = None
        self.loaded = False
        self.status = "not_loaded"

    def load(self) -> None:
        if self.loaded:
            return
        self.loaded = True

        if not self.model_path.exists():
            self.status = f"missing: {self.model_path}"
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            self.status = "ultralytics_not_installed"
            return

        self.model = YOLO(str(self.model_path))
        self.status = "loaded"

    def predict(self, image: Image.Image) -> tuple[list[SegmentPrediction], list[str]]:
        self.load()
        if self.model is None:
            return self._fallback(image), [f"{self.name} недоступен ({self.status})."]

        kwargs: dict[str, Any] = {
            "source": np.array(image.convert("RGB")),
            "imgsz": self.image_size,
            "conf": self.confidence,
            "retina_masks": True,
            "verbose": False,
        }
        if self.device and self.device != "auto":
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        if not results:
            return [], [f"{self.name} не вернул результат."]

        result = results[0]
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or getattr(masks, "data", None) is None:
            return [], [f"{self.name} не вернул маски."]

        mask_array = _tensor_to_numpy(masks.data)
        classes = _tensor_to_numpy(getattr(boxes, "cls", [])) if boxes is not None else []
        confidences = _tensor_to_numpy(getattr(boxes, "conf", [])) if boxes is not None else []
        names = getattr(result, "names", None) or getattr(self.model, "names", {})

        segments: list[SegmentPrediction] = []
        for index, raw_mask in enumerate(mask_array):
            label_index = int(classes[index]) if index < len(classes) else 0
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            label = str(names.get(label_index, label_index))
            mask = _resize_mask(raw_mask, image.size)
            area_fraction = float(mask.mean())
            if area_fraction <= 0:
                continue
            segments.append(
                SegmentPrediction(
                    label=label,
                    confidence=confidence,
                    area_fraction=area_fraction,
                    mask=mask,
                )
            )

        if not segments:
            return [], [f"{self.name} вернул пустые маски."]
        return segments, []

    def _fallback(self, image: Image.Image) -> list[SegmentPrediction]:
        if self.fallback_kind == "plate":
            mask = elliptical_plate_mask(image.size)
            return [
                SegmentPrediction(
                    label="plate",
                    confidence=0.0,
                    area_fraction=float(mask.mean()),
                    density_group="ignore",
                    use_for_mass=False,
                    mask=mask,
                )
            ]

        mask = foreground_mask(image)
        return [
            SegmentPrediction(
                label="food_region",
                confidence=0.0,
                area_fraction=float(mask.mean()),
                density_group="unknown",
                use_for_mass=False,
                mask=mask,
            )
        ]


def elliptical_plate_mask(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    y, x = np.ogrid[:height, :width]
    center_x = width / 2.0
    center_y = height / 2.0
    radius_x = width * 0.38
    radius_y = height * 0.34
    return ((x - center_x) ** 2 / radius_x**2 + (y - center_y) ** 2 / radius_y**2) <= 1.0


def _resize_mask(mask: Any, size: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask, dtype=np.float32)
    array = (array > 0.5).astype("uint8") * 255
    image = Image.fromarray(array, mode="L").resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)

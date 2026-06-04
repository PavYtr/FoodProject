from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from food_project.class_mapping import ClassMapping
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
        self.last_stats: dict[str, float] = self._empty_stats()

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
            segments = self._fallback(image)
            self.last_stats = self._stats_from_segments(segments, raw_count=len(segments))
            return segments, [f"{self.name} unavailable ({self.status})."]

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
            self.last_stats = self._empty_stats()
            return [], [f"{self.name} returned no result."]

        result = results[0]
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or getattr(masks, "data", None) is None:
            self.last_stats = self._empty_stats()
            return [], [f"{self.name} returned no instance masks."]

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
                    metadata={"model_class_id": label_index},
                )
            )

        self.last_stats = self._stats_from_confidences(
            confidences=confidences,
            raw_count=len(mask_array),
            kept_count=len(segments),
        )
        if not segments:
            return [], [f"{self.name} returned empty masks."]
        return segments, []

    @staticmethod
    def _empty_stats() -> dict[str, float]:
        return {
            "n_masks_raw": 0.0,
            "n_masks_kept": 0.0,
            "seg_conf_mean": 0.0,
            "seg_conf_max": 0.0,
        }

    @classmethod
    def _stats_from_segments(
        cls,
        segments: list[SegmentPrediction],
        raw_count: int,
    ) -> dict[str, float]:
        return cls._stats_from_confidences(
            confidences=[segment.confidence for segment in segments],
            raw_count=raw_count,
            kept_count=len(segments),
        )

    @staticmethod
    def _stats_from_confidences(
        confidences: Any,
        raw_count: int,
        kept_count: int,
    ) -> dict[str, float]:
        values = np.asarray(confidences, dtype=np.float32)
        return {
            "n_masks_raw": float(raw_count),
            "n_masks_kept": float(kept_count),
            "seg_conf_mean": float(values.mean()) if values.size else 0.0,
            "seg_conf_max": float(values.max()) if values.size else 0.0,
            "n_plate_masks": float(raw_count),
            "plate_conf_mean": float(values.mean()) if values.size else 0.0,
            "plate_conf_max": float(values.max()) if values.size else 0.0,
        }

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


class YOLOSemanticSegmentationModel(YOLOSegmentationModel):
    def __init__(
        self,
        name: str,
        model_path: Path,
        mapping: ClassMapping,
        device: str = "cpu",
        image_size: int = 640,
    ) -> None:
        super().__init__(
            name=name,
            model_path=model_path,
            device=device,
            image_size=image_size,
            confidence=0.0,
            fallback_kind="food",
        )
        self.mapping = mapping
        self.last_stats = self._semantic_stats(raw_count=0, kept_count=0)

    def predict(self, image: Image.Image) -> tuple[list[SegmentPrediction], list[str]]:
        self.load()
        if self.model is None:
            segments = self._fallback(image)
            kept_count = sum(1 for segment in segments if segment.use_for_mass)
            self.last_stats = self._semantic_stats(raw_count=len(segments), kept_count=kept_count)
            return segments, [f"{self.name} unavailable ({self.status})."]

        image_array = np.array(image.convert("RGB"))
        height, width = image_array.shape[:2]
        kwargs: dict[str, Any] = {
            "source": image_array,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device and self.device != "auto":
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        if not results:
            self.last_stats = self._semantic_stats(raw_count=0, kept_count=0)
            return [], [f"{self.name} returned no result."]

        result = results[0]
        semantic_mask = getattr(result, "semantic_mask", None)
        if semantic_mask is None or getattr(semantic_mask, "data", None) is None:
            self.last_stats = self._semantic_stats(raw_count=0, kept_count=0)
            return [], [f"{self.name} returned no semantic_mask."]

        semantic_map = _tensor_to_numpy(semantic_mask.data).astype(np.int32)
        semantic_map = np.squeeze(semantic_map)
        if semantic_map.ndim != 2:
            self.last_stats = self._semantic_stats(raw_count=0, kept_count=0)
            return [], [f"{self.name} returned semantic_mask with unsupported shape {semantic_map.shape}."]
        if semantic_map.shape[:2] != (height, width):
            semantic_map = _resize_label_map(semantic_map, (height, width))

        class_ids, counts = np.unique(semantic_map, return_counts=True)
        segments: list[SegmentPrediction] = []
        image_area = float(max(height * width, 1))
        names = getattr(result, "names", None) or getattr(self.model, "names", {})

        for model_class_id, area in zip(class_ids.tolist(), counts.tolist()):
            label = str(names.get(int(model_class_id), model_class_id))
            if not self.mapping.use_segment_for_mass(label):
                continue

            area_px = int(area)
            if area_px <= 0:
                continue

            segments.append(
                SegmentPrediction(
                    label=label,
                    confidence=0.0,
                    area_fraction=float(area_px / image_area),
                    density_group=self.mapping.density_group_for_foodseg(label),
                    use_for_mass=True,
                    mask=semantic_map == int(model_class_id),
                    metadata={
                        "class_id": int(model_class_id),
                        "model_class_id": int(model_class_id),
                    },
                )
            )

        self.last_stats = self._semantic_stats(raw_count=len(class_ids), kept_count=len(segments))
        if not segments:
            return [], [f"{self.name} returned no usable semantic food classes."]
        return segments, []

    @staticmethod
    def _semantic_stats(raw_count: int, kept_count: int) -> dict[str, float]:
        return {
            "n_masks_raw": float(raw_count),
            "n_masks_kept": float(kept_count),
            "n_semantic_classes_raw": float(raw_count),
            "n_semantic_classes_kept": float(kept_count),
            "seg_conf_mean": 0.0,
            "seg_conf_max": 0.0,
        }


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


def _resize_label_map(label_map: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = shape_hw
    try:
        import cv2

        return cv2.resize(
            label_map.astype(np.uint16),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    except Exception:
        image = Image.fromarray(label_map.astype(np.uint16), mode="I;16").resize(
            (width, height),
            Image.Resampling.NEAREST,
        )
        return np.asarray(image, dtype=np.int32)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)

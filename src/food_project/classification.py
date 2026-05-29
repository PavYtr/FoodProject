from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from food_project.schemas import ClassificationPrediction


class YOLOClassifier:
    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        image_size: int = 640,
        top_k: int = 5,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.top_k = top_k
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

    def predict(self, image: Image.Image) -> tuple[list[ClassificationPrediction], list[str]]:
        self.load()
        if self.model is None:
            return (
                [ClassificationPrediction("unknown", 0.0)],
                [f"Классификатор недоступен ({self.status})."],
            )

        kwargs: dict[str, Any] = {
            "source": np.array(image.convert("RGB")),
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device and self.device != "auto":
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        if not results:
            return [ClassificationPrediction("unknown", 0.0)], ["Классификатор не вернул результат."]

        result = results[0]
        probabilities = getattr(result, "probs", None)
        if probabilities is None:
            return [ClassificationPrediction("unknown", 0.0)], ["В результате YOLO нет probs для классификации."]

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        indices = _as_list(getattr(probabilities, "top5", []))[: self.top_k]
        confidences = _as_list(getattr(probabilities, "top5conf", []))[: self.top_k]

        predictions: list[ClassificationPrediction] = []
        for index, confidence in zip(indices, confidences):
            int_index = int(_scalar(index))
            label = str(names.get(int_index, int_index))
            predictions.append(ClassificationPrediction(label, float(_scalar(confidence))))

        if not predictions:
            return [ClassificationPrediction("unknown", 0.0)], ["Классификатор вернул пустой top-k."]
        return predictions, []


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "cpu"):
        value = value.cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)

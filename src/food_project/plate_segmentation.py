from __future__ import annotations

from pathlib import Path

from food_project.food_segmentation import YOLOSegmentationModel
from food_project.schemas import SegmentPrediction


class PlateSegmentationModel(YOLOSegmentationModel):
    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        image_size: int = 640,
        confidence: float = 0.25,
    ) -> None:
        super().__init__(
            name="Сегментатор тарелки",
            model_path=model_path,
            device=device,
            image_size=image_size,
            confidence=confidence,
            fallback_kind="plate",
        )

    @staticmethod
    def select_plate(segments: list[SegmentPrediction]) -> SegmentPrediction | None:
        if not segments:
            return None
        plate = max(segments, key=lambda item: item.area_fraction)
        plate.density_group = "ignore"
        plate.use_for_mass = False
        return plate

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from food_project.pipeline import FoodNutritionPipeline  # noqa: E402
from food_project.visualization import depth_to_image, overlay_segments  # noqa: E402


@lru_cache(maxsize=1)
def get_pipeline() -> FoodNutritionPipeline:
    pipeline = FoodNutritionPipeline()
    pipeline.load_models()
    return pipeline


def predict_for_ui(
    image: Any,
    text: str | None,
    show_intermediate: bool,
) -> tuple[Any, ...]:
    if image is None:
        return _empty_outputs("Загрузите изображение блюда.")

    try:
        result = get_pipeline().predict(image=image, text=text)
    except Exception as exc:  # pragma: no cover - UI safety net
        return _empty_outputs(f"Ошибка inference: {exc}")

    food_overlay = None
    plate_overlay = None
    depth_image = None
    if show_intermediate:
        food_overlay = overlay_segments(result.source_image, result.food_segments)
        plate_overlay = (
            overlay_segments(result.source_image, [result.plate_segment])
            if result.plate_segment
            else result.source_image
        )
        depth_image = depth_to_image(result.depth_map)

    status_rows = [[name, status] for name, status in result.model_status.items()]

    return (
        result.summary_text(),
        result.classification_rows(),
        result.nutrition.as_rows(),
        food_overlay,
        plate_overlay,
        depth_image,
        result.segment_rows(),
        result.feature_rows(),
        result.warnings_text(),
        status_rows,
    )


def _empty_outputs(message: str) -> tuple[Any, ...]:
    return (
        message,
        [],
        [],
        None,
        None,
        None,
        [],
        [],
        message,
        [],
    )

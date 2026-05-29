from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from food_project.config import PipelineConfig  # noqa: E402
from food_project.mass_estimation import FALLBACK_FEATURE_NAMES  # noqa: E402
from food_project.pipeline import FoodNutritionPipeline  # noqa: E402


def test_pipeline_runs_without_model_weights() -> None:
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((70, 45, 250, 205), fill=(240, 240, 235), outline=(170, 170, 165), width=4)
    draw.rectangle((115, 90, 205, 150), fill=(210, 90, 70))

    pipeline = FoodNutritionPipeline(config=_missing_models_config())
    result = pipeline.predict(image)

    assert result.dish_class
    assert result.food_segments
    assert result.food_segments[0].use_for_mass is False
    assert result.plate_segment is not None
    assert result.nutrition.source in {"catboost", "heuristic", "unavailable"}


def test_try2_feature_schema_contains_notebook_columns() -> None:
    assert len(FALLBACK_FEATURE_NAMES) == 185
    assert FALLBACK_FEATURE_NAMES[:10] == [
        "n_masks_raw",
        "n_masks_kept",
        "seg_conf_mean",
        "seg_conf_max",
        "n_plate_masks",
        "plate_conf_mean",
        "plate_conf_max",
        "image_h",
        "image_w",
        "image_area_px",
    ]
    assert FALLBACK_FEATURE_NAMES[-6:] == [
        "food101_top3",
        "food101_top3_conf",
        "food101_top4",
        "food101_top4_conf",
        "food101_top5",
        "food101_top5_conf",
    ]


def _missing_models_config() -> PipelineConfig:
    missing_dir = ROOT / "__missing_models__"
    return PipelineConfig(
        root_dir=ROOT,
        models_dir=missing_dir,
        classifier_model=missing_dir / "yolo_cls.pt",
        food_segmentation_model=missing_dir / "yolo_food_seg.pt",
        plate_segmentation_model=missing_dir / "plate_seg.pt",
        mass_model=missing_dir / "mass_model.cbm",
        food101_mapping=ROOT / "class_mappings" / "food101_dish_groups.csv",
        foodseg103_mapping=ROOT / "class_mappings" / "foodseg103_density_groups.csv",
        depth_model="heuristic",
    )

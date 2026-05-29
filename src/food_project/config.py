from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError("PyYAML is required to read configs/app.yaml") from exc

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded or {}


def _as_path(value: str | Path | None, default: str | Path) -> Path:
    return Path(value) if value else Path(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class PipelineConfig:
    root_dir: Path
    models_dir: Path
    classifier_model: Path
    food_segmentation_model: Path
    plate_segmentation_model: Path
    mass_model: Path
    food101_mapping: Path
    foodseg103_mapping: Path
    device: str = "cpu"
    image_size: int = 640
    depth_model: str = "depth-anything-v3-base"
    classifier_top_k: int = 5
    class_confidence_threshold: float = 0.10
    mask_confidence_threshold: float = 0.25
    plate_diameter_cm: float = 26.0
    default_food_height_cm: float = 1.8
    allow_heuristic_mass_fallback: bool = False
    mass_model_output_transform: str = "log1p"

    def resolve_path(self, value: Path) -> Path:
        return value if value.is_absolute() else self.root_dir / value

    @property
    def resolved_models_dir(self) -> Path:
        return self.resolve_path(self.models_dir)

    @property
    def resolved_classifier_model(self) -> Path:
        return self.resolve_path(self.classifier_model)

    @property
    def resolved_food_segmentation_model(self) -> Path:
        return self.resolve_path(self.food_segmentation_model)

    @property
    def resolved_plate_segmentation_model(self) -> Path:
        return self.resolve_path(self.plate_segmentation_model)

    @property
    def resolved_mass_model(self) -> Path:
        return self.resolve_path(self.mass_model)

    @property
    def resolved_food101_mapping(self) -> Path:
        return self.resolve_path(self.food101_mapping)

    @property
    def resolved_foodseg103_mapping(self) -> Path:
        return self.resolve_path(self.foodseg103_mapping)


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    requested_config = (
        os.getenv("FOOD_PROJECT_CONFIG")
        or config_path
        or REPO_ROOT / "configs" / "app.yaml"
    )
    raw = _read_yaml(Path(requested_config))

    paths = raw.get("paths", {})
    inference = raw.get("inference", {})

    root_dir = Path(os.getenv("FOOD_PROJECT_ROOT") or raw.get("root_dir") or REPO_ROOT)
    models_dir = _as_path(
        os.getenv("FOOD_PROJECT_MODELS_DIR") or paths.get("models_dir"),
        "models",
    )

    return PipelineConfig(
        root_dir=root_dir,
        models_dir=models_dir,
        classifier_model=_as_path(
            os.getenv("FOOD_PROJECT_CLASSIFIER_MODEL") or paths.get("classifier_model"),
            models_dir / "yolo_cls.pt",
        ),
        food_segmentation_model=_as_path(
            os.getenv("FOOD_PROJECT_FOOD_SEG_MODEL")
            or paths.get("food_segmentation_model"),
            models_dir / "yolo_food_seg.pt",
        ),
        plate_segmentation_model=_as_path(
            os.getenv("FOOD_PROJECT_PLATE_SEG_MODEL")
            or paths.get("plate_segmentation_model"),
            models_dir / "plate_seg.pt",
        ),
        mass_model=_as_path(
            os.getenv("FOOD_PROJECT_MASS_MODEL") or paths.get("mass_model"),
            models_dir / "mass_model.cbm",
        ),
        food101_mapping=_as_path(
            paths.get("food101_mapping"),
            "class_mappings/food101_dish_groups.csv",
        ),
        foodseg103_mapping=_as_path(
            paths.get("foodseg103_mapping"),
            "class_mappings/foodseg103_density_groups.csv",
        ),
        device=os.getenv("FOOD_PROJECT_DEVICE") or inference.get("device", "cpu"),
        image_size=_as_int(inference.get("image_size"), 640),
        depth_model=str(
            os.getenv("DEPTH_MODEL")
            or os.getenv("FOOD_PROJECT_DEPTH_MODEL")
            or inference.get("depth_model")
            or "depth-anything-v3-base"
        ),
        classifier_top_k=_as_int(inference.get("classifier_top_k"), 5),
        class_confidence_threshold=_as_float(
            inference.get("class_confidence_threshold"),
            0.10,
        ),
        mask_confidence_threshold=_as_float(
            inference.get("mask_confidence_threshold"),
            0.25,
        ),
        plate_diameter_cm=_as_float(inference.get("plate_diameter_cm"), 26.0),
        default_food_height_cm=_as_float(
            inference.get("default_food_height_cm"),
            1.8,
        ),
        allow_heuristic_mass_fallback=_as_bool(
            os.getenv("FOOD_PROJECT_ALLOW_HEURISTIC_MASS_FALLBACK")
            or inference.get("allow_heuristic_mass_fallback"),
            False,
        ),
        mass_model_output_transform=str(
            os.getenv("FOOD_PROJECT_MASS_MODEL_OUTPUT_TRANSFORM")
            or inference.get("mass_model_output_transform")
            or "log1p"
        ),
    )

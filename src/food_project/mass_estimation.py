from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from food_project.class_mapping import ClassMapping, DENSITY_G_PER_CM3
from food_project.schemas import ClassificationPrediction, NutritionEstimate, SegmentPrediction


MASS_GROUPS = [
    "bread",
    "dairy_dessert",
    "dairy_fat",
    "dessert",
    "fruit",
    "fruit_dried",
    "fruit_fat",
    "leafy_veg",
    "liquid",
    "meat",
    "meat_processed",
    "mixed_main",
    "mushroom",
    "nuts",
    "protein",
    "sauce",
    "seafood",
    "seaweed",
    "starch",
    "starch_legume",
    "starch_main",
    "starch_veg",
    "unknown",
    "vegetable",
]


def _notebook_feature_names() -> list[str]:
    names = [
        "n_masks_raw",
        "n_masks_kept",
        "n_semantic_classes_raw",
        "n_semantic_classes_kept",
        "seg_conf_mean",
        "seg_conf_max",
        "n_plate_masks",
        "plate_conf_mean",
        "plate_conf_max",
        "image_h",
        "image_w",
        "image_area_px",
    ]
    for group in MASS_GROUPS:
        names.extend([f"area_group_{group}", f"count_group_{group}", f"pvol_group_{group}"])
    names.extend(
        [
            "area_px",
            "sqrt_area",
            "log_area",
            "area_ratio",
            "n_unique_seg_classes",
            "union_area_px",
            "union_perimeter",
            "union_compactness",
            "union_solidity",
            "union_equiv_diameter",
            "plate_area_px",
            "plate_perimeter",
            "plate_compactness",
            "plate_solidity",
            "plate_equiv_diameter",
            "plate_area_ratio",
            "food_to_plate_area_ratio",
            "food_plate_intersection_px",
            "food_in_plate_ratio",
            "plate_covered_by_food_ratio",
            "top1_seg_class",
            "top1_density_group",
            "top1_area_ratio",
            "top2_seg_class",
            "top2_density_group",
            "top2_area_ratio",
            "top3_seg_class",
            "top3_density_group",
            "top3_area_ratio",
            "dominant_seg_class",
            "dominant_density_group",
        ]
    )
    for method in ("p05p95", "iqrz"):
        for ring in ("015", "035", "070"):
            for metric in (
                "plate_depth",
                "vol_plate_minus_food",
                "vol_food_minus_plate",
                "mean_abs_height",
                "p75_abs_height",
                "p95_abs_height",
            ):
                names.append(f"union_{method}_ring{ring}_{metric}")
    names.extend([f"area_ratio_group_{group}" for group in MASS_GROUPS])
    names.extend(
        [
            "food101_top1",
            "food101_top1_conf",
            "food101_dish_group",
            "food101_entropy",
            "food101_top2",
            "food101_top2_conf",
            "food101_top3",
            "food101_top3_conf",
            "food101_top4",
            "food101_top4_conf",
            "food101_top5",
            "food101_top5_conf",
        ]
    )
    return names


FALLBACK_FEATURE_NAMES = _notebook_feature_names()


class MassEstimator:
    def __init__(
        self,
        mapping: ClassMapping,
        plate_diameter_cm: float = 26.0,
        default_food_height_cm: float = 1.8,
        mass_model_path: Path | None = None,
        allow_heuristic_fallback: bool = False,
        output_transform: str = "log1p",
    ) -> None:
        self.mapping = mapping
        self.plate_diameter_cm = plate_diameter_cm
        self.default_food_height_cm = default_food_height_cm
        self.mass_model_path = mass_model_path
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.output_transform = output_transform
        self.mass_model: Any | None = None
        self.mass_model_loaded = False
        self.status = "not_configured" if mass_model_path is None else "not_loaded"

    def load_model(self) -> None:
        self._load_mass_model()

    def estimate(
        self,
        dish_group: str,
        top_classes: list[ClassificationPrediction],
        food_segments: list[SegmentPrediction],
        plate_segment: SegmentPrediction | None,
        plate_segments: list[SegmentPrediction],
        depth_map: Any,
        depth_stats: dict[str, float],
        image_size: tuple[int, int],
        food_segmentation_stats: dict[str, float] | None = None,
        plate_segmentation_stats: dict[str, float] | None = None,
    ) -> tuple[NutritionEstimate, dict[str, float], list[str]]:
        feature_bank, ui_features, warnings = self._build_feature_bank(
            dish_group=dish_group,
            top_classes=top_classes,
            food_segments=food_segments,
            food_segmentation_stats=food_segmentation_stats,
            plate_segment=plate_segment,
            plate_segments=plate_segments,
            plate_segmentation_stats=plate_segmentation_stats,
            depth_map=depth_map,
            depth_stats=depth_stats,
            image_size=image_size,
        )

        fallback_mass_g = float(feature_bank["heuristic_mass_g"])
        mass_g, source, model_warning = self._predict_with_mass_model(
            feature_bank=feature_bank,
            fallback_mass_g=fallback_mass_g,
        )
        if model_warning:
            warnings.append(model_warning)

        ui_features["mass_model_used"] = 1.0 if source == "catboost" else 0.0
        ui_features["heuristic_fallback_used"] = 1.0 if source == "heuristic" else 0.0

        if source == "unavailable":
            return (
                NutritionEstimate(source="unavailable"),
                ui_features,
                warnings,
            )

        proteins_100g, fats_100g, carbs_100g = self.mapping.macros_for_group(dish_group)
        proteins_g = mass_g * proteins_100g / 100.0
        fats_g = mass_g * fats_100g / 100.0
        carbs_g = mass_g * carbs_100g / 100.0
        calories = proteins_g * 4.0 + fats_g * 9.0 + carbs_g * 4.0

        return (
            NutritionEstimate(
                mass_g=mass_g,
                proteins_g=proteins_g,
                fats_g=fats_g,
                carbs_g=carbs_g,
                calories_kcal=calories,
                source=source,
            ),
            ui_features,
            warnings,
        )

    def _build_feature_bank(
        self,
        dish_group: str,
        top_classes: list[ClassificationPrediction],
        food_segments: list[SegmentPrediction],
        plate_segment: SegmentPrediction | None,
        plate_segments: list[SegmentPrediction],
        depth_map: Any,
        depth_stats: dict[str, float],
        image_size: tuple[int, int],
        food_segmentation_stats: dict[str, float] | None = None,
        plate_segmentation_stats: dict[str, float] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float], list[str]]:
        warnings: list[str] = []
        width, height = image_size
        image_area_px = float(max(width * height, 1))
        shape = (height, width)
        food_segmentation_stats = food_segmentation_stats or {}
        plate_segmentation_stats = plate_segmentation_stats or {}

        usable_segments = [segment for segment in food_segments if segment.use_for_mass]
        if not usable_segments:
            warnings.append("Food masks are empty; CatBoost mass prediction may be unreliable.")

        food_masks = [_mask_for_segment(segment, shape) for segment in usable_segments]
        plate_masks = [_mask_for_segment(segment, shape) for segment in plate_segments]
        plate_mask = _union_mask(plate_masks, shape)
        if not plate_mask.any() and plate_segment:
            plate_mask = _mask_for_segment(plate_segment, shape)
        union_mask = _union_mask(food_masks, shape)

        union_area_px = float(union_mask.sum())
        plate_area_px = float(plate_mask.sum())
        intersection_px = float((union_mask & plate_mask).sum())

        area_ratio = union_area_px / image_area_px
        plate_area_ratio = plate_area_px / image_area_px
        food_to_plate_area_ratio = union_area_px / plate_area_px if plate_area_px else 0.0
        food_in_plate_ratio = intersection_px / max(union_area_px, 1.0)
        plate_covered_by_food_ratio = intersection_px / max(plate_area_px, 1.0)

        depth = _resize_depth(depth_map, shape)
        depth_mean = float(depth_stats.get("depth_mean", float(depth.mean())))
        depth_std = float(depth_stats.get("depth_std", float(depth.std())))

        density = self._weighted_density(usable_segments, dish_group)
        plate_radius_cm = self.plate_diameter_cm / 2.0
        plate_area_cm2 = math.pi * plate_radius_cm**2
        height_cm = self.default_food_height_cm * (0.75 + depth_mean)
        food_area_cm2 = min(food_to_plate_area_ratio, 1.0) * plate_area_cm2
        heuristic_mass_g = food_area_cm2 * height_cm * density

        bank: dict[str, Any] = {
            "image_h": float(height),
            "image_w": float(width),
            "image_area_px": image_area_px,
            "area_px": union_area_px,
            "area_ratio": area_ratio,
            "sqrt_area": math.sqrt(max(union_area_px, 0.0)),
            "log_area": math.log1p(max(union_area_px, 0.0)),
            "union_area_px": union_area_px,
            "plate_area_px": plate_area_px,
            "plate_area_ratio": plate_area_ratio,
            "food_plate_intersection_px": intersection_px,
            "food_to_plate_area_ratio": food_to_plate_area_ratio,
            "food_in_plate_ratio": food_in_plate_ratio,
            "plate_covered_by_food_ratio": plate_covered_by_food_ratio,
            "n_masks_raw": _stat(food_segmentation_stats, "n_masks_raw", len(food_segments)),
            "n_masks_kept": _stat(food_segmentation_stats, "n_masks_kept", len(usable_segments)),
            "n_semantic_classes_raw": _stat(
                food_segmentation_stats,
                "n_semantic_classes_raw",
                _stat(food_segmentation_stats, "n_masks_raw", len(food_segments)),
            ),
            "n_semantic_classes_kept": _stat(
                food_segmentation_stats,
                "n_semantic_classes_kept",
                _stat(food_segmentation_stats, "n_masks_kept", len(usable_segments)),
            ),
            "n_plate_masks": _stat(plate_segmentation_stats, "n_plate_masks", len(plate_segments)),
            "n_unique_seg_classes": float(len(_segment_class_ids(usable_segments))),
            "seg_conf_mean": _stat(
                food_segmentation_stats,
                "seg_conf_mean",
                _mean([segment.confidence for segment in food_segments]),
            ),
            "seg_conf_max": _stat(
                food_segmentation_stats,
                "seg_conf_max",
                _max([segment.confidence for segment in food_segments]),
            ),
            "plate_conf_mean": _stat(
                plate_segmentation_stats,
                "plate_conf_mean",
                _mean([segment.confidence for segment in plate_segments]),
            ),
            "plate_conf_max": _stat(
                plate_segmentation_stats,
                "plate_conf_max",
                _max([segment.confidence for segment in plate_segments]),
            ),
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "depth_min": float(depth_stats.get("depth_min", float(depth.min()))),
            "depth_max": float(depth_stats.get("depth_max", float(depth.max()))),
            "height_cm": height_cm,
            "density_g_cm3": density,
            "heuristic_mass_g": heuristic_mass_g,
        }

        bank.update(_shape_features("union", union_mask))
        bank.update(_shape_features("plate", plate_mask))
        bank.update(self._classification_features(top_classes, dish_group))
        bank.update(self._segment_rank_features(usable_segments, image_area_px, union_area_px))
        bank.update(self._group_features(usable_segments, depth, image_area_px, union_area_px))
        bank.update(_depth_union_features(depth, union_mask, plate_mask))

        ui_features = {
            key: float(value)
            for key, value in bank.items()
            if isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))
        }
        return bank, ui_features, warnings

    def _classification_features(
        self,
        top_classes: list[ClassificationPrediction],
        dish_group: str,
    ) -> dict[str, Any]:
        features: dict[str, Any] = {"food101_dish_group": dish_group}
        probabilities = [max(item.confidence, 0.0) for item in top_classes[:5]]
        total = sum(probabilities)
        entropy = 0.0
        if total > 0:
            normalized = [probability / total for probability in probabilities]
            entropy = -sum(probability * math.log(probability + 1e-9) for probability in normalized)

        features["food101_entropy"] = entropy
        for index in range(5):
            item = top_classes[index] if index < len(top_classes) else None
            rank = index + 1
            features[f"food101_top{rank}"] = item.label if item else "unknown"
            features[f"food101_top{rank}_conf"] = float(item.confidence) if item else 0.0
        return features

    def _segment_rank_features(
        self,
        segments: list[SegmentPrediction],
        image_area_px: float,
        union_area_px: float,
    ) -> dict[str, Any]:
        sorted_segments = sorted(segments, key=lambda segment: _segment_area_px(segment, image_area_px), reverse=True)
        features: dict[str, Any] = {}
        for index in range(3):
            segment = sorted_segments[index] if index < len(sorted_segments) else None
            rank = index + 1
            features[f"top{rank}_seg_class"] = segment.label if segment else "unknown"
            features[f"top{rank}_density_group"] = segment.density_group if segment else "unknown"
            features[f"top{rank}_area_ratio"] = (
                _segment_area_px(segment, image_area_px) / max(union_area_px, 1.0)
                if segment
                else 0.0
            )

        dominant = sorted_segments[0] if sorted_segments else None
        features["dominant_seg_class"] = dominant.label if dominant else "unknown"
        features["dominant_density_group"] = dominant.density_group if dominant else "unknown"
        return features

    def _group_features(
        self,
        segments: list[SegmentPrediction],
        depth: np.ndarray,
        image_area_px: float,
        union_area_px: float,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        depth_versions = _normalize_depth_versions(depth)
        p05p95 = depth_versions["p05p95"]
        for group in MASS_GROUPS:
            features[f"area_group_{group}"] = 0.0
            features[f"area_ratio_group_{group}"] = 0.0
            features[f"count_group_{group}"] = 0.0
            features[f"pvol_group_{group}"] = 0.0

        for segment in segments:
            group = segment.density_group if segment.density_group in MASS_GROUPS else "unknown"
            mask = _mask_for_segment(segment, depth.shape) if segment.mask is not None else None
            area_px = float(mask.sum()) if mask is not None else segment.area_fraction * image_area_px

            features[f"area_group_{group}"] += area_px
            features[f"count_group_{group}"] += 1.0
            if mask is not None and mask.any():
                height_features = _height_features(mask, p05p95, "tmp")
                features[f"pvol_group_{group}"] += (
                    height_features.get("tmp_ring035_vol_plate_minus_food", 0.0)
                    + height_features.get("tmp_ring035_vol_food_minus_plate", 0.0)
                )

        for group in MASS_GROUPS:
            features[f"area_ratio_group_{group}"] = features[f"area_group_{group}"] / max(union_area_px, 1.0)

        return features

    def _load_mass_model(self) -> None:
        if self.mass_model_loaded:
            return
        self.mass_model_loaded = True

        if self.mass_model_path is None:
            self.status = "not_configured"
            return
        if not self.mass_model_path.exists():
            self.status = f"missing: {self.mass_model_path}"
            return

        try:
            from catboost import CatBoostRegressor
        except ImportError:
            self.status = "catboost_not_installed"
            return

        try:
            model = CatBoostRegressor()
            model.load_model(str(self.mass_model_path))
        except Exception as exc:  # pragma: no cover - depends on external model file
            self.status = f"load_error: {exc}"
            return

        self.mass_model = model
        self.status = "loaded"

    def _predict_with_mass_model(
        self,
        feature_bank: dict[str, Any],
        fallback_mass_g: float,
    ) -> tuple[float, str, str | None]:
        self._load_mass_model()
        if self.mass_model is None:
            message = f"CatBoost mass model is unavailable ({self.status}); mass was not predicted."
            if self.allow_heuristic_fallback:
                return fallback_mass_g, "heuristic", f"{message} Explicit heuristic fallback is enabled."
            return 0.0, "unavailable", message

        expected_names = self._expected_feature_names()
        row = {
            name: feature_bank.get(name, _default_feature_value(name))
            for name in expected_names
        }

        try:
            import pandas as pd
            from catboost import Pool

            frame = pd.DataFrame([row], columns=expected_names)
            cat_features = self._cat_feature_indices(frame)
            data = Pool(frame, cat_features=cat_features or None)
            raw_prediction = float(self.mass_model.predict(data)[0])
            prediction = self._postprocess_prediction(raw_prediction)
        except Exception as exc:  # pragma: no cover - depends on external model schema
            self.status = f"predict_error: {exc}"
            message = f"CatBoost mass model predict failed: {exc}"
            if self.allow_heuristic_fallback:
                return fallback_mass_g, "heuristic", f"{message}. Explicit heuristic fallback is enabled."
            return 0.0, "unavailable", message

        if not math.isfinite(prediction) or prediction <= 0:
            self.status = "predict_invalid"
            message = "CatBoost mass model returned an invalid mass value."
            if self.allow_heuristic_fallback:
                return fallback_mass_g, "heuristic", f"{message} Explicit heuristic fallback is enabled."
            return 0.0, "unavailable", message

        self.status = "loaded_used"
        return min(prediction, 5000.0), "catboost", None

    def _postprocess_prediction(self, raw_prediction: float) -> float:
        transform = self.output_transform.strip().lower()
        if transform in {"log1p", "log1p_mass", "expm1"}:
            return float(np.expm1(raw_prediction))
        if transform in {"none", "identity", "mass"}:
            return raw_prediction
        return float(np.expm1(raw_prediction))

    def _expected_feature_names(self) -> list[str]:
        if self.mass_model is None:
            return FALLBACK_FEATURE_NAMES

        names = list(getattr(self.mass_model, "feature_names_", []) or [])
        if names and any(names):
            return [name or f"feature_{index}" for index, name in enumerate(names)]

        try:
            feature_count = int(self.mass_model.get_feature_count())
        except Exception:  # pragma: no cover - depends on catboost model object
            feature_count = len(FALLBACK_FEATURE_NAMES)

        if feature_count <= len(FALLBACK_FEATURE_NAMES):
            return FALLBACK_FEATURE_NAMES[:feature_count]
        return FALLBACK_FEATURE_NAMES + [
            f"feature_{index}" for index in range(len(FALLBACK_FEATURE_NAMES), feature_count)
        ]

    def _cat_feature_indices(self, frame: Any) -> list[int]:
        try:
            indices = list(self.mass_model.get_cat_feature_indices())
        except Exception:  # pragma: no cover - depends on catboost model object
            indices = []

        if indices:
            for index in indices:
                column = frame.columns[index]
                frame[column] = frame[column].astype(str)
            return indices

        return [
            index
            for index, value in enumerate(frame.iloc[0].tolist())
            if isinstance(value, str)
        ]

    def _weighted_density(self, segments: list[SegmentPrediction], dish_group: str) -> float:
        weighted_sum = 0.0
        area_sum = 0.0
        for segment in segments:
            group = segment.density_group if segment.density_group != "unknown" else dish_group
            density = DENSITY_G_PER_CM3.get(group, DENSITY_G_PER_CM3["unknown"])
            weighted_sum += density * segment.area_fraction
            area_sum += segment.area_fraction

        if area_sum <= 0:
            return self.mapping.density_for_group(dish_group)
        return weighted_sum / area_sum


def _mask_for_segment(segment: SegmentPrediction | None, shape: tuple[int, int]) -> np.ndarray:
    if segment is None or segment.mask is None:
        return np.zeros(shape, dtype=bool)

    mask = np.asarray(segment.mask, dtype=bool)
    if mask.shape == shape:
        return mask

    try:
        from PIL import Image

        resized = Image.fromarray(mask.astype("uint8") * 255, mode="L").resize(
            (shape[1], shape[0]),
            Image.Resampling.NEAREST,
        )
        return np.asarray(resized) > 127
    except Exception:
        return np.zeros(shape, dtype=bool)


def _union_mask(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for mask in masks:
        union |= mask
    return union


def _segment_area_px(segment: SegmentPrediction, image_area_px: float) -> float:
    if segment.mask is not None:
        return float(np.asarray(segment.mask, dtype=bool).sum())
    return float(segment.area_fraction * image_area_px)


def _segment_class_ids(segments: list[SegmentPrediction]) -> set[Any]:
    class_ids: set[Any] = set()
    for segment in segments:
        class_ids.add(segment.metadata.get("class_id", segment.label))
    return class_ids


def _stat(stats: dict[str, float], name: str, default: float | int) -> float:
    try:
        value = float(stats.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _resize_depth(depth_map: Any, shape: tuple[int, int]) -> np.ndarray:
    if depth_map is None:
        return np.zeros(shape, dtype=np.float32)

    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.shape != shape:
        try:
            import cv2

            depth = cv2.resize(
                depth,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        except Exception:
            try:
                from PIL import Image

                image = Image.fromarray(depth, mode="F").resize(
                    (shape[1], shape[0]),
                    Image.Resampling.BILINEAR,
                )
                depth = np.asarray(image, dtype=np.float32)
            except Exception:
                depth = np.zeros(shape, dtype=np.float32)

    return np.asarray(depth, dtype=np.float32)


def _shape_features(prefix: str, mask: np.ndarray) -> dict[str, float]:
    area = float(mask.sum())
    perimeter = _perimeter(mask)
    compactness = 0.0 if perimeter <= 0 else area / max(perimeter**2, 1e-6)
    equiv_diameter = math.sqrt((4.0 * area) / math.pi) if area > 0 else 0.0
    return {
        f"{prefix}_perimeter": perimeter,
        f"{prefix}_compactness": compactness,
        f"{prefix}_solidity": _solidity(mask),
        f"{prefix}_equiv_diameter": equiv_diameter,
    }


def _perimeter(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0

    try:
        import cv2

        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return float(sum(cv2.arcLength(contour, True) for contour in contours))
    except Exception:
        padded = np.pad(mask, 1)
        center = padded[1:-1, 1:-1]
        edge = center & (
            ~padded[:-2, 1:-1]
            | ~padded[2:, 1:-1]
            | ~padded[1:-1, :-2]
            | ~padded[1:-1, 2:]
        )
        return float(edge.sum())


def _solidity(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0

    try:
        import cv2

        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = 0.0
        for contour in contours:
            if len(contour) >= 3:
                hull_area += float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area <= 0:
            return 0.0
        return float(mask.sum()) / hull_area
    except Exception:
        return 1.0


def _depth_union_features(
    depth: np.ndarray,
    union_mask: np.ndarray,
    plate_mask: np.ndarray,
) -> dict[str, float]:
    features: dict[str, float] = {}
    del plate_mask
    depth_versions = _normalize_depth_versions(depth)
    features.update(_height_features(union_mask, depth_versions["p05p95"], "union_p05p95"))
    features.update(_height_features(union_mask, depth_versions["iqr_z"], "union_iqrz"))
    return features


def _height_features(
    mask: np.ndarray,
    depth_norm: np.ndarray,
    prefix: str,
    ring_fracs: tuple[float, ...] = (0.015, 0.035, 0.07),
) -> dict[str, float]:
    height, width = mask.shape[:2]
    features: dict[str, float] = {}
    food_values = depth_norm[mask]
    food_values = food_values[np.isfinite(food_values)]
    if food_values.size == 0:
        for fraction in ring_fracs:
            tag = f"{prefix}_ring{int(fraction * 1000):03d}"
            features[f"{tag}_plate_depth"] = 0.0
            features[f"{tag}_vol_plate_minus_food"] = 0.0
            features[f"{tag}_vol_food_minus_plate"] = 0.0
            features[f"{tag}_mean_abs_height"] = 0.0
            features[f"{tag}_p75_abs_height"] = 0.0
            features[f"{tag}_p95_abs_height"] = 0.0
        return features

    finite_depth = depth_norm[np.isfinite(depth_norm)]
    fallback_depth = float(np.median(finite_depth)) if finite_depth.size else 0.0
    for fraction in ring_fracs:
        kernel_size = max(5, int(round(min(height, width) * fraction)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        try:
            import cv2

            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            dilated = cv2.dilate(mask.astype("uint8"), kernel, iterations=1).astype(bool)
            ring = dilated & ~mask
        except Exception:
            ring = ~mask

        ring_values = depth_norm[ring]
        ring_values = ring_values[np.isfinite(ring_values)]
        if ring_values.size < 20:
            ring_values = depth_norm[~mask]
            ring_values = ring_values[np.isfinite(ring_values)]

        plate_depth = float(np.median(ring_values)) if ring_values.size else fallback_depth
        plate_minus_food = np.clip(plate_depth - food_values, 0, None)
        food_minus_plate = np.clip(food_values - plate_depth, 0, None)
        if plate_minus_food.size:
            plate_minus_food = np.clip(plate_minus_food, 0, np.percentile(plate_minus_food, 95))
        if food_minus_plate.size:
            food_minus_plate = np.clip(food_minus_plate, 0, np.percentile(food_minus_plate, 95))

        abs_height = np.abs(food_values - plate_depth)
        tag = f"{prefix}_ring{int(fraction * 1000):03d}"
        features[f"{tag}_plate_depth"] = plate_depth
        features[f"{tag}_vol_plate_minus_food"] = float(plate_minus_food.sum())
        features[f"{tag}_vol_food_minus_plate"] = float(food_minus_plate.sum())
        features[f"{tag}_mean_abs_height"] = float(abs_height.mean())
        features[f"{tag}_p75_abs_height"] = float(np.percentile(abs_height, 75))
        features[f"{tag}_p95_abs_height"] = float(np.percentile(abs_height, 95))

    return features


def _normalize_depth_versions(depth: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(depth, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        zeros = np.zeros_like(values, dtype=np.float32)
        return {"p05p95": zeros, "iqr_z": zeros}

    p05, p25, p50, p75, p95 = np.percentile(finite, [5, 25, 50, 75, 95])
    p05p95_scale = max(float(p95 - p05), 1e-6)
    iqr_scale = max(float(p75 - p25), 1e-6)
    p05p95 = np.clip((values - p05) / p05p95_scale, 0.0, 1.0).astype(np.float32)
    iqr_z = np.clip((values - p50) / iqr_scale, -5.0, 5.0).astype(np.float32)
    return {"p05p95": p05p95, "iqr_z": iqr_z}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _default_feature_value(name: str) -> Any:
    return "unknown" if _is_categorical_feature(name) else 0.0


def _is_categorical_feature(name: str) -> bool:
    if name.endswith("_group") or name.endswith("_class"):
        return True
    if name.startswith("food101_top") and not name.endswith("_conf"):
        return True
    return name in {"food101_dish_group", "dominant_density_group", "dominant_seg_class"}

from __future__ import annotations

from pathlib import Path
from typing import Any

from food_project.class_mapping import ClassMapping
from food_project.classification import YOLOClassifier
from food_project.config import PipelineConfig, load_config
from food_project.depth import DepthEstimator
from food_project.food_segmentation import YOLOSemanticSegmentationModel
from food_project.mass_estimation import MassEstimator
from food_project.plate_segmentation import PlateSegmentationModel
from food_project.preprocessing import ensure_pil_image, quality_warnings
from food_project.schemas import PredictionResult


class FoodNutritionPipeline:
    def __init__(self, config: PipelineConfig | None = None, config_path: str | Path | None = None) -> None:
        self.config = config or load_config(config_path)
        self.mapping = ClassMapping(
            self.config.resolved_food101_mapping,
            self.config.resolved_foodseg103_mapping,
        )
        self.classifier = YOLOClassifier(
            model_path=self.config.resolved_classifier_model,
            device=self.config.device,
            image_size=self.config.image_size,
            top_k=self.config.classifier_top_k,
        )
        self.food_segmenter = YOLOSemanticSegmentationModel(
            name="Food semantic segmenter",
            model_path=self.config.resolved_food_segmentation_model,
            mapping=self.mapping,
            device=self.config.device,
            image_size=self.config.image_size,
        )
        self.plate_segmenter = PlateSegmentationModel(
            model_path=self.config.resolved_plate_segmentation_model,
            device=self.config.device,
            image_size=self.config.image_size,
            confidence=self.config.mask_confidence_threshold,
        )
        self.depth_estimator = DepthEstimator(model_id=self.config.depth_model)
        self.mass_estimator = MassEstimator(
            mapping=self.mapping,
            plate_diameter_cm=self.config.plate_diameter_cm,
            default_food_height_cm=self.config.default_food_height_cm,
            mass_model_path=self.config.resolved_mass_model,
            allow_heuristic_fallback=self.config.allow_heuristic_mass_fallback,
            output_transform=self.config.mass_model_output_transform,
        )

    def load_models(self) -> dict[str, str]:
        self.classifier.load()
        self.food_segmenter.load()
        self.plate_segmenter.load()
        self.mass_estimator.load_model()
        return self._model_status()

    def predict(self, image: Any) -> PredictionResult:
        pil_image = ensure_pil_image(image)
        warnings = quality_warnings(pil_image)

        top_classes, classifier_warnings = self.classifier.predict(pil_image)
        warnings.extend(classifier_warnings)

        dish_class = top_classes[0].label
        class_confidence = top_classes[0].confidence
        dish_group = self.mapping.dish_group_for_food101(dish_class)

        food_segments, food_warnings = self.food_segmenter.predict(pil_image)
        warnings.extend(food_warnings)
        for segment in food_segments:
            segment.density_group = self.mapping.density_group_for_foodseg(segment.label)
            segment.use_for_mass = segment.use_for_mass and self.mapping.use_segment_for_mass(segment.label)

        plate_segments, plate_warnings = self.plate_segmenter.predict(pil_image)
        warnings.extend(plate_warnings)
        plate_segment = self.plate_segmenter.select_plate(plate_segments)

        depth_map, depth_stats, depth_warnings = self.depth_estimator.predict(pil_image)
        warnings.extend(depth_warnings)

        nutrition, feature_values, mass_warnings = self.mass_estimator.estimate(
            dish_group=dish_group,
            top_classes=top_classes,
            food_segments=food_segments,
            food_segmentation_stats=self.food_segmenter.last_stats,
            plate_segment=plate_segment,
            plate_segments=plate_segments,
            plate_segmentation_stats=self.plate_segmenter.last_stats,
            depth_map=depth_map,
            depth_stats=depth_stats,
            image_size=pil_image.size,
        )
        warnings.extend(mass_warnings)

        feature_values.update(depth_stats)

        return PredictionResult(
            dish_class=dish_class,
            class_confidence=class_confidence,
            dish_group=dish_group,
            top_classes=top_classes,
            food_segments=food_segments,
            plate_segment=plate_segment,
            depth_map=depth_map,
            depth_stats=depth_stats,
            nutrition=nutrition,
            warnings=_deduplicate(warnings),
            model_status=self._model_status(),
            feature_values=feature_values,
            source_image=pil_image,
        )

    def visualize_result(self, result: PredictionResult) -> dict[str, Any]:
        from food_project.visualization import depth_to_image, overlay_segments

        food_overlay = overlay_segments(result.source_image, result.food_segments)
        plate_overlay = (
            overlay_segments(result.source_image, [result.plate_segment])
            if result.plate_segment
            else result.source_image
        )
        return {
            "food_overlay": food_overlay,
            "plate_overlay": plate_overlay,
            "depth_map": depth_to_image(result.depth_map),
        }

    def _model_status(self) -> dict[str, str]:
        return {
            "classifier": self.classifier.status,
            "food_segmentation": self.food_segmenter.status,
            "plate_segmentation": self.plate_segmenter.status,
            "depth": self.depth_estimator.status,
            "mass_model": self.mass_estimator.status,
        }


def _deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

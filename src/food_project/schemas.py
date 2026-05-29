from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClassificationPrediction:
    label: str
    confidence: float

    def as_row(self) -> list[Any]:
        return [self.label, round(self.confidence, 4)]


@dataclass
class SegmentPrediction:
    label: str
    confidence: float
    area_fraction: float
    density_group: str = "unknown"
    use_for_mass: bool = True
    mask: Any = field(default=None, repr=False)

    def as_row(self) -> list[Any]:
        return [
            self.label,
            round(self.confidence, 4),
            round(self.area_fraction, 4),
            self.density_group,
            self.use_for_mass,
        ]


@dataclass
class NutritionEstimate:
    mass_g: float = 0.0
    proteins_g: float = 0.0
    fats_g: float = 0.0
    carbs_g: float = 0.0
    calories_kcal: float = 0.0
    source: str = "unavailable"

    def as_rows(self) -> list[list[Any]]:
        if self.source == "unavailable":
            return [
                ["Масса", "не рассчитана", ""],
                ["Белки", "не рассчитаны", ""],
                ["Жиры", "не рассчитаны", ""],
                ["Углеводы", "не рассчитаны", ""],
                ["Калории", "не рассчитаны", ""],
                ["Источник", "CatBoost недоступен", ""],
            ]

        return [
            ["Масса", round(self.mass_g, 1), "г"],
            ["Белки", round(self.proteins_g, 1), "г"],
            ["Жиры", round(self.fats_g, 1), "г"],
            ["Углеводы", round(self.carbs_g, 1), "г"],
            ["Калории", round(self.calories_kcal, 1), "ккал"],
            ["Источник", self.source, ""],
        ]


@dataclass
class PredictionResult:
    dish_class: str
    class_confidence: float
    dish_group: str
    top_classes: list[ClassificationPrediction]
    food_segments: list[SegmentPrediction]
    plate_segment: SegmentPrediction | None
    depth_map: Any
    depth_stats: dict[str, float]
    nutrition: NutritionEstimate
    warnings: list[str]
    model_status: dict[str, str]
    feature_values: dict[str, Any]
    source_image: Any = field(default=None, repr=False)

    def classification_rows(self) -> list[list[Any]]:
        return [item.as_row() for item in self.top_classes]

    def segment_rows(self) -> list[list[Any]]:
        rows = [item.as_row() for item in self.food_segments]
        if self.plate_segment:
            rows.append(self.plate_segment.as_row())
        return rows

    def feature_rows(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for key, value in self.feature_values.items():
            if isinstance(value, (int, float)):
                rows.append([key, round(value, 4)])
            else:
                rows.append([key, value])
        return rows

    def warnings_text(self) -> str:
        return "\n".join(f"- {warning}" for warning in self.warnings) or "-"

    def summary_text(self) -> str:
        confidence = round(self.class_confidence * 100, 1)
        if self.nutrition.source == "unavailable":
            mass_line = "Масса: не рассчитана"
            calories_line = "Калории: не рассчитаны"
        else:
            mass_line = f"Масса: {round(self.nutrition.mass_g, 1)} г"
            calories_line = f"Калории: {round(self.nutrition.calories_kcal, 1)} ккал"

        return (
            f"Класс: {self.dish_class}\n"
            f"Группа: {self.dish_group}\n"
            f"Уверенность: {confidence}%\n"
            f"{mass_line}\n"
            f"{calories_line}\n"
            f"Модель массы: {self.nutrition.source}"
        )

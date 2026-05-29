from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


DENSITY_G_PER_CM3 = {
    "ignore": 0.0,
    "unknown": 0.85,
    "liquid": 1.0,
    "soup_liquid": 1.0,
    "sauce": 0.95,
    "dessert": 0.70,
    "dairy_dessert": 0.75,
    "dairy_fat": 0.95,
    "bread": 0.35,
    "breakfast": 0.55,
    "starch": 0.65,
    "starch_main": 0.78,
    "starch_legume": 0.78,
    "starch_veg": 0.72,
    "mixed_main": 0.80,
    "sandwich": 0.62,
    "salad": 0.35,
    "leafy_veg": 0.25,
    "vegetable": 0.55,
    "fruit": 0.62,
    "fruit_dried": 0.90,
    "fruit_fat": 0.88,
    "meat": 1.05,
    "meat_main": 1.02,
    "meat_processed": 1.00,
    "seafood": 0.92,
    "protein": 0.86,
    "nuts": 0.65,
    "mushroom": 0.35,
    "seaweed": 0.45,
}


MACROS_PER_100G = {
    "unknown": (8.0, 8.0, 16.0),
    "dessert": (5.0, 16.0, 45.0),
    "dairy_dessert": (4.0, 10.0, 28.0),
    "dairy_fat": (18.0, 28.0, 3.0),
    "bread": (8.0, 4.0, 48.0),
    "breakfast": (10.0, 12.0, 28.0),
    "starch": (3.0, 12.0, 35.0),
    "starch_main": (8.0, 8.0, 32.0),
    "starch_legume": (10.0, 6.0, 28.0),
    "mixed_main": (12.0, 14.0, 26.0),
    "sandwich": (13.0, 13.0, 25.0),
    "salad": (3.0, 7.0, 10.0),
    "leafy_veg": (2.0, 1.0, 5.0),
    "vegetable": (2.0, 2.0, 8.0),
    "fruit": (1.0, 0.3, 13.0),
    "fruit_dried": (2.0, 0.5, 65.0),
    "fruit_fat": (2.0, 15.0, 9.0),
    "meat": (26.0, 16.0, 0.0),
    "meat_main": (24.0, 18.0, 3.0),
    "meat_processed": (18.0, 24.0, 2.0),
    "seafood": (21.0, 7.0, 3.0),
    "protein": (13.0, 10.0, 2.0),
    "soup_liquid": (4.0, 3.0, 8.0),
    "liquid": (1.0, 1.0, 10.0),
    "sauce": (2.0, 18.0, 8.0),
    "nuts": (18.0, 52.0, 18.0),
}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class ClassMapping:
    def __init__(self, food101_path: Path, foodseg103_path: Path) -> None:
        self.food101_groups: dict[str, str] = {}
        self.foodseg_density_groups: dict[str, str] = {}
        self.foodseg_use_for_mask: dict[str, bool] = {}
        self.load_food101(food101_path)
        self.load_foodseg103(foodseg103_path)

    def load_food101(self, path: Path) -> None:
        if not path.exists():
            return
        for row in _read_csv(path):
            class_name = _first_value(row, "class_name", "food101_class", "label", "name")
            dish_group = _first_value(row, "dish_group", "group", "category")
            if class_name and dish_group:
                self.food101_groups[normalize_label(class_name)] = dish_group.strip()

    def load_foodseg103(self, path: Path) -> None:
        if not path.exists():
            return
        for row in _read_csv(path):
            class_name = _first_value(row, "class_name", "foodseg_class", "label", "name")
            density_group = _first_value(row, "density_group", "dish_group", "group")
            use_for_mask = row.get("use_for_mask", "true")
            if class_name and density_group:
                key = normalize_label(class_name)
                self.foodseg_density_groups[key] = density_group.strip()
                self.foodseg_use_for_mask[key] = _truthy(use_for_mask)

    def dish_group_for_food101(self, label: str) -> str:
        return self.food101_groups.get(normalize_label(label), "unknown")

    def density_group_for_foodseg(self, label: str) -> str:
        key = normalize_label(label)
        return self.foodseg_density_groups.get(key, self.food101_groups.get(key, "unknown"))

    def use_segment_for_mass(self, label: str) -> bool:
        return self.foodseg_use_for_mask.get(normalize_label(label), True)

    def density_for_group(self, group: str) -> float:
        return DENSITY_G_PER_CM3.get(group, DENSITY_G_PER_CM3["unknown"])

    def macros_for_group(self, group: str) -> tuple[float, float, float]:
        return MACROS_PER_100G.get(group, MACROS_PER_100G["unknown"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return ""

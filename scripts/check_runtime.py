from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODEL_PATH = ROOT / "models" / "mass_model.cbm"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from food_project.mass_estimation import FALLBACK_FEATURE_NAMES  # noqa: E402


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    if sys.version_info >= (3, 14):
        print("error: CatBoost is not available for this project on Python 3.14+")
        print("use Python 3.11 or 3.12, or run the Docker image")
        return 1

    missing_packages = [
        name
        for name in ("catboost", "pandas", "ultralytics", "depth_estimation")
        if importlib.util.find_spec(name) is None
    ]
    if missing_packages:
        print("error: missing packages: " + ", ".join(missing_packages))
        return 1

    from catboost import CatBoostRegressor

    if not MODEL_PATH.exists():
        print(f"error: missing model: {MODEL_PATH}")
        return 1

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    feature_count = int(model.get_feature_count())
    cat_indices = list(model.get_cat_feature_indices())

    print(f"mass model: {MODEL_PATH}")
    print(f"feature count: {feature_count}")
    print(f"notebook feature count: {len(FALLBACK_FEATURE_NAMES)}")
    print(f"categorical feature indices: {cat_indices}")

    if feature_count != len(FALLBACK_FEATURE_NAMES):
        print("error: CatBoost feature count does not match pseudo-depth-v3-try2 notebook schema")
        return 1

    print("runtime ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

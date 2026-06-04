# FoodProject App

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CatBoost](https://img.shields.io/badge/Regression-CatBoost-FFCC00)
![YOLO](https://img.shields.io/badge/Vision-Ultralytics%20YOLO-111111)
![Depth Anything](https://img.shields.io/badge/Depth-Depth%20Anything-6A5ACD)
![Docker](https://img.shields.io/badge/Runtime-Docker-2496ED?logo=docker&logoColor=white)


FoodProject App is a Gradio inference application for estimating food mass and basic nutrition from a dish image. The pipeline combines dish classification, food and plate segmentation, pseudo-depth estimation, CatBoost-based mass regression, and dish-group nutrition mapping.

The application is intended for local inference and demos. Results are model estimates, not medical or dietary advice.

## What It Does

- Classifies the dish with a YOLO Food101 classifier.
- Segments visible food regions with a YOLO FoodSeg103 model.
- Detects the plate to estimate scale and food-to-plate ratios.
- Builds a pseudo-depth map with DepthAnything V3 when available, or a deterministic heuristic fallback.
- Builds mass-estimation features.
- Predicts food mass with a CatBoost regressor.
- Estimates proteins, fats, carbohydrates, and calories from the predicted mass and dish group.
- Shows intermediate masks, depth maps, feature values, warnings, and model status in the Gradio UI.

## Pipeline

```text
input image
  -> Food101 classification
  -> FoodSeg103 food segmentation
  -> plate segmentation
  -> pseudo-depth estimation
  -> feature builder
  -> CatBoost mass regression
  -> nutrition estimate
  -> Gradio UI
```

### Runtime Flow

1. The user uploads an image.
2. `app/inference.py` loads and caches `FoodNutritionPipeline`.
3. The classifier predicts top-k Food101 dish classes.
4. Food and plate segmenters produce masks and area ratios.
5. The depth estimator creates a pseudo-depth map or falls back to a heuristic map.
6. `MassEstimator` builds the CatBoost feature row from classification, segmentation, plate geometry, and depth statistics.
7. The CatBoost model predicts mass; macro estimates are derived from dish-group mappings.
8. The UI displays the summary, nutrition table, masks, depth map, model status, warnings, and numeric features.

## Model Metrics

The current inference feature schema mirrors `notebooks/train_catboost_pseudodepth_v3_with_semantic.ipynb`.
The semantic validation metrics are stored in `models/mass_model_semantic.json`.
To reproduce those metrics, `models/mass_model.cbm` must be the CatBoost export trained with the semantic feature schema.

| Metric | Value |
| --- | ---: |
| Validation MAE | 63.78 g |
| Validation RMSE | 93.44 g |
| Validation R2 | 0.6480 |
| Baseline median MAE | 123.34 g |
| Train / validation rows | 2595 / 649 |
| Feature count | 187 |


## Repository Structure

```text
app/
  app.py                 Gradio UI entry point
  inference.py           UI adapter around the inference pipeline
src/food_project/
  pipeline.py            End-to-end FoodNutritionPipeline
  classification.py      YOLO Food101 classifier wrapper
  food_segmentation.py   YOLO FoodSeg103 semantic and instance segmentation wrappers
  plate_segmentation.py  Plate segmentation wrapper
  depth.py               DepthAnything V3 integration and heuristic fallback
  mass_estimation.py     Feature builder and CatBoost mass inference
  class_mapping.py       Dish, density, and macro mappings
  schemas.py             Result dataclasses
configs/
  app.yaml               Default model paths and inference settings
class_mappings/
  food101_dish_groups.csv
  foodseg103_density_groups.csv
models/
  yolo_cls.pt
  yolo_food_sem.pt
  plate_seg.pt
  mass_model.cbm
  mass_model_metrics.json
docker/
  Dockerfile
tests/
  test_pipeline_smoke.py
```

## Requirements

- `git`
- Python 3.11, 3.12, or 3.13. Python 3.11 is recommended.
- `pip`
- The model files listed in `models/README.md`
- `Docker`, if you want to run the containerized version

CatBoost is not expected to work with Python 3.14+ for this project.

## Setup

Clone the repository:

```bash
git clone https://github.com/PavYtr/FoodProject.git
cd FoodProject
```

Build and start the application:

```bash
docker compose up --build
```

Open:

```text
http://localhost:7860
```

The Compose service mounts `models/` and `class_mappings/` as read-only volumes, so you can replace model files without rebuilding the image.

## Configuration

The default configuration file is:

```text
configs/app.yaml
```

Important settings:

| Setting | Purpose |
| --- | --- |
| `paths.classifier_model` | YOLO Food101 classifier weights |
| `paths.food_segmentation_model` | YOLO FoodSeg103 semantic segmentation weights |
| `paths.plate_segmentation_model` | Plate segmentation weights |
| `paths.mass_model` | CatBoost mass regressor |
| `inference.device` | Inference device, for example `cpu` or `cuda` |
| `inference.image_size` | YOLO inference image size |
| `inference.depth_model` | Depth model id, or `heuristic` to disable external depth inference |
| `inference.plate_diameter_cm` | Reference plate diameter used for scale estimation |
| `inference.allow_heuristic_mass_fallback` | Enables heuristic mass fallback when CatBoost is unavailable |

Supported environment overrides:

```text
FOOD_PROJECT_CONFIG
FOOD_PROJECT_ROOT
FOOD_PROJECT_MODELS_DIR
FOOD_PROJECT_CLASSIFIER_MODEL
FOOD_PROJECT_FOOD_SEG_MODEL
FOOD_PROJECT_PLATE_SEG_MODEL
FOOD_PROJECT_MASS_MODEL
FOOD_PROJECT_DEVICE
DEPTH_MODEL
FOOD_PROJECT_DEPTH_MODEL
FOOD_PROJECT_ALLOW_HEURISTIC_MASS_FALLBACK
FOOD_PROJECT_MASS_MODEL_OUTPUT_TRANSFORM
GRADIO_SERVER_NAME
GRADIO_SERVER_PORT
```

`.env.example` documents common values, but the application does not automatically load `.env` files. Export variables in your shell or pass them through Docker Compose.

## Model Files

Expected files:

| File | Role |
| --- | --- |
| `models/yolo_cls.pt` | Food101 dish classifier |
| `models/yolo_food_sem.pt` | Default FoodSeg103 semantic segmentation model |
| `models/plate_seg.pt` | Plate segmentation model |
| `models/mass_model.cbm` | CatBoost mass regressor |
| `models/mass_model_metrics.json` | Semantic notebook metrics and feature schema |


The segmentation components can start without weights and use deterministic demo-mask fallbacks. The classifier reports `unknown` when its weights are unavailable. Mass prediction is stricter: if `mass_model.cbm` or CatBoost is unavailable, mass and nutrition are reported as unavailable unless `FOOD_PROJECT_ALLOW_HEURISTIC_MASS_FALLBACK=true` is explicitly enabled.

## Notes and Limitations

- Estimates depend heavily on image quality, visible plate geometry, segmentation quality, and the configured plate diameter.
- Depth inference can fall back to a heuristic map if the external depth model cannot be loaded.
- Nutrition values are derived from dish-group macro mappings, not from ingredient-level recognition.
- For GPU inference, install a compatible PyTorch build and set `FOOD_PROJECT_DEVICE=cuda`.

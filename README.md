# FoodProject App


## Пайплайн

```text
image
  -> YOLO Food101 classifier
  -> YOLO FoodSeg103 segmentation
  -> YOLO plate segmentation
  -> DepthAnything V3 pseudo-depth
  -> notebook-compatible feature builder
  -> CatBoost mass regression
  -> mass and nutrition estimate
  -> Gradio UI
```


Метрики notebook-модели:

| Metric | Value |
| --- | ---: |
| valid MAE | 78.62 g |
| valid RMSE | 112.27 g |
| valid R2 | 0.4919 |
| baseline median MAE | 123.34 g |
| train / valid rows | 2595 / 649 |

## Структура

```text
app/
  app.py                 # Gradio UI
  inference.py           # UI adapter
src/food_project/
  pipeline.py            # единый FoodNutritionPipeline
  classification.py      # YOLO Food101 wrapper
  food_segmentation.py   # YOLO FoodSeg103 wrapper
  plate_segmentation.py  # YOLO plate wrapper
  depth.py               # DepthAnything V3 / heuristic fallback
  mass_estimation.py     # признаки try2 и CatBoost inference
configs/
  app.yaml               # пути, device, thresholds, depth model
models/
  yolo_cls.pt
  yolo_food_seg.pt
  plate_seg.pt
  mass_model.cbm
docker/
  Dockerfile
```

## Docker

```bash
docker compose up --build
```

Открой `http://localhost:7860`.

## Конфигурация

Основной конфиг: `configs/app.yaml`.

Поддерживаются переменные окружения:

```text
FOOD_PROJECT_CONFIG
FOOD_PROJECT_MODELS_DIR
FOOD_PROJECT_CLASSIFIER_MODEL
FOOD_PROJECT_FOOD_SEG_MODEL
FOOD_PROJECT_PLATE_SEG_MODEL
FOOD_PROJECT_MASS_MODEL
FOOD_PROJECT_ALLOW_HEURISTIC_MASS_FALLBACK
FOOD_PROJECT_DEVICE
DEPTH_MODEL
FOOD_PROJECT_DEPTH_MODEL
GRADIO_SERVER_NAME
GRADIO_SERVER_PORT
```


# FoodProject App

Сервисный inference для пайплайна из ноутбука
`FoodProject/mass_estimation/pseudo-depth-v3-try2/train_catboost_pseudodepth_v3_try2.ipynb`.

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

CatBoost модель `models/mass_model.cbm` совпадает с
`FoodProject/models/catboost_pseudodepth_v3_try2.cbm`. Предикт модели обучен на
`log1p(mass)`, приложение делает обратное преобразование через `expm1`.

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

## Локальный запуск

CatBoost не поддерживается в текущем Python 3.14 окружении. Для локального
запуска используй Python 3.11/3.12 или Docker.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_runtime.py
python app/app.py
```

Открой `http://127.0.0.1:7860`.

## Docker

```bash
docker compose up --build
```

Открой `http://localhost:7860`.

В контейнере Gradio слушает `0.0.0.0:7860`. Папки `models/` и
`class_mappings/` монтируются read-only, поэтому один и тот же образ можно
поднять на другом устройстве при наличии этих файлов.

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

Если CatBoost или `mass_model.cbm` недоступны, приложение не подменяет результат
формулой: в UI будет `Масса: не рассчитана`. Эвристический fallback можно
включить только явно через `FOOD_PROJECT_ALLOW_HEURISTIC_MASS_FALLBACK=true`;
это demo-режим, не режим метрик ноутбука.

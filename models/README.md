# Models

Expected inference weights:

| File | Role |
| --- | --- |
| `yolo_cls.pt` | Food101 dish classifier |
| `yolo_food_seg.pt` | FoodSeg103 food segmentation model |
| `plate_seg.pt` | Plate segmentation model |
| `mass_model.cbm` | Primary CatBoost mass regressor |
| `mass_model_metrics.json` | Metrics summary from pseudo-depth-v3-try2 notebook |

The YOLO parts can start without weights and fall back to deterministic demo masks. Mass estimation is different: `mass_model.cbm` is the primary predictor. If CatBoost or this file is unavailable, mass is reported as not calculated unless `FOOD_PROJECT_ALLOW_HEURISTIC_MASS_FALLBACK=true` is explicitly enabled.

`mass_model.cbm` is the same model as `FoodProject/models/catboost_pseudodepth_v3_try2.cbm`.

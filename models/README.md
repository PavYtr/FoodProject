# Models

Expected inference weights:

| File | Role |
| --- | --- |
| `yolo_cls.pt` | Food101 dish classifier |
| `yolo_food_seg.pt` | FoodSeg103 food segmentation model |
| `plate_seg.pt` | Plate segmentation model |
| `mass_model.cbm` | Primary CatBoost mass regressor |
| `mass_model_metrics.json` | Metrics summary |


`mass_model.cbm` is the same model as `FoodProject/models/catboost_pseudodepth_v3_try2.cbm`.

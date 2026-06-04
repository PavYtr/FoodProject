# Models

Expected inference weights:

| File | Role |
| --- | --- |
| `yolo_cls.pt` | Food101 dish classifier |
| `yolo_food_sem.pt` | Default FoodSeg103 semantic segmentation model |
| `plate_seg.pt` | Plate segmentation model |
| `mass_model.cbm` | Primary CatBoost mass regressor |
| `mass_model_semantic.json` | Semantic notebook metrics and feature schema |


To reproduce the metrics from `notebooks/train_catboost_pseudodepth_v3_with_semantic.ipynb`,
`mass_model.cbm` must be the CatBoost export trained with `yolo_food_sem.pt` semantic features.

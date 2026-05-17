import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth_estimation import pipeline

class DepthEstimationModule:
    def __init__(self, model_id=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = model_id or os.getenv("DEPTH_MODEL", "depth-anything-v3-base")

        self.pipe = pipeline(
            "depth-estimation",
            model=model_id,
            device=str(self.device),
        )

    def get_depth_matrix(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        with torch.no_grad():
            result = self.pipe(image)

        return np.asarray(result.depth, dtype=np.float32)


# Backward-compatible name for existing imports.
DepthAnything3Module = DepthEstimationModule


if __name__ == "__main__":
    engine = DepthEstimationModule()
    print("Depth estimation module loaded successfully.")

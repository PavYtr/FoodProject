import os
import torch
import numpy as np
from PIL import Image

from depth_anything_3.api import DepthAnything3

class DepthAnything3Module:
    def __init__(self, model_id=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = model_id or os.getenv("MODEL_PATH", "depth-anything/DA3-SMALL")

        self.model = DepthAnything3.from_pretrained(model_id)
        self.model = self.model.to(device=self.device)
        self.model.eval()

    def get_depth_matrix(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        img_array = np.array(image)

        with torch.no_grad():
            pred = self.model.inference([img_array])
        
        depth_matrix = pred.depth[0]
        return depth_matrix

if __name__ == "__main__":
    engine = DepthAnything3Module()
    print("Модуль DepthAnything3 успешно загружен.")

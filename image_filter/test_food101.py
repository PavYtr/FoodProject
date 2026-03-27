import os
import numpy as np
import cv2

from pathlib import Path
from tqdm import tqdm
from filter import analyze_image

def main(dataset_path):
    root = Path(dataset_path) / "images"
    report_file = "food101_defects_report.txt"

    images = list(root.rglob("*.jpg"))[:1200]

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Path | Defects\n")
        f.write("-" * 50 + "\n")

        for image in tqdm(images):
            rel_path = image.relative_to(dataset_path)
            defects = analyze_image(image)
            if defects:
                line = f"{rel_path} | {",".join(defects)}"
                f.write(line + "\n")


if __name__ == "__main__":
    main("C:/training/Food101")

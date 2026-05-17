import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from depth_module import DepthAnything3Module


def normalize_depth(depth):
    depth = np.asarray(depth)
    depth_min = float(np.nanmin(depth))
    depth_max = float(np.nanmax(depth))

    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)

    normalized = (depth - depth_min) / (depth_max - depth_min)
    return (normalized * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--npy", default="depth.npy", help="Path for the raw depth matrix")
    parser.add_argument("--png", default="depth.png", help="Path for the normalized depth preview")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    engine = DepthAnything3Module()
    depth = engine.get_depth_matrix(str(image_path))
    depth = np.asarray(depth)

    np.save(args.npy, depth)
    Image.fromarray(normalize_depth(depth)).save(args.png)

    print(f"Depth shape: {depth.shape}")
    print(f"Depth min/max: {float(np.nanmin(depth)):.6f} / {float(np.nanmax(depth)):.6f}")
    print(f"Saved matrix: {args.npy}")
    print(f"Saved preview: {args.png}")


if __name__ == "__main__":
    main()

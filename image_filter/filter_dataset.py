import os
from pathlib import Path
from filter import analyze_image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

def create_filtered_list(image_dir, output_file, use_multiprocessing=True, num_workers=None):
    image_paths = list(Path(image_dir).rglob("*.jpg"))
    cleaned_paths = []
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Filtering {len(image_paths)} images in {image_dir}...")

    if use_multiprocessing:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(analyze_image, p): p for p in image_paths}
            for future in tqdm(as_completed(futures), total=len(image_paths), desc="Analyzing images", unit="img"):
                p = futures[future]
                if future.result() is None:
                    cleaned_paths.append(str(p.absolute()))
    else:
        for p in tqdm(image_paths, desc="Analyzing images", unit="img"):
            if analyze_image(p) is None:
                cleaned_paths.append(str(p.absolute()))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned_paths))

    print("Done. Cleaned list saved to:", output_file)

if __name__ == "__main__":
    create_filtered_list("C:/training/Food101/images", "food101_cleaned_list.txt")
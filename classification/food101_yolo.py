import os
import shutil
from pathlib import Path


txt_file = 'C:/training/image_filter/food101_cleaned_list.txt'
output_dir = Path('yolo_food101_ready')
train_ratio = 0.8

with open(txt_file, 'r') as f:
    paths = [Path(line.strip()) for line in f if line.strip()]

data = {}
for p in paths:
    class_name = p.parent.name
    if class_name not in data:
        data[class_name] = []
    data[class_name].append(p)

for class_name, imgs in data.items():
    import random
    random.shuffle(imgs)
    
    split = int(len(imgs) * train_ratio)
    train_imgs = imgs[:split]
    val_imgs = imgs[split:]

    for split_name, split_list in [('train', train_imgs), ('val', val_imgs)]:
        class_path = output_dir / split_name / class_name
        class_path.mkdir(parents=True, exist_ok=True)
        
        for img_p in split_list:
            target = class_path / img_p.name
            try:
                if not target.exists():
                    os.symlink(img_p.absolute(), target)
            except OSError:
                shutil.copy2(img_p, target)

print(f"Готово! {output_dir.absolute()}")
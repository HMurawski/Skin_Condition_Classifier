
import random
import shutil
from pathlib import Path


data_dir = Path("data/train")
val_dir = Path("data/val")
val_ratio = 0.2  # 20% val

random.seed(42)


for class_dir in data_dir.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    images = list(class_dir.glob("*.*"))  # .jpg, .png, etc.
    random.shuffle(images)

    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]

    
    val_class_dir = val_dir / class_name
    val_class_dir.mkdir(parents=True, exist_ok=True)

    
    for img_path in val_images:
        shutil.move(str(img_path), val_class_dir / img_path.name)

print("✅ Done")

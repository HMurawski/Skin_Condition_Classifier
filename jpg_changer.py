import os
from PIL import Image


folders = [
    "data/train/healthy",
    "data/val/healthy"
]

for folder in folders:
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jfif"):
            old_path = os.path.join(folder, filename)
            new_name = filename.replace(".jfif", ".jpg")
            new_path = os.path.join(folder, new_name)

            try:
                with Image.open(old_path) as img:
                    rgb_img = img.convert("RGB")  # JFIF do RGB
                    rgb_img.save(new_path, "JPEG")
                    print(f" {filename} → {new_name}")
                
                os.remove(old_path)  # delete .jfif
            except Exception as e:
                print(f"FAILED: {filename}: {e}")

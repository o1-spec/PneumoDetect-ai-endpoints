import os
import shutil
import random

SOURCE_DIR = "dataset/processed"
OUTPUT_DIR = "dataset/final"
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)

for class_name in ["NORMAL", "PNEUMONIA"]:
    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = [
        f for f in os.listdir(source_class_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]
    train_images = images[val_count:]

    for split, split_images in [("train", train_images), ("val", val_images)]:
        target_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(target_dir, exist_ok=True)

        for image in split_images:
            src = os.path.join(source_class_dir, image)
            dst = os.path.join(target_dir, image)
            shutil.copy(src, dst)

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")

print("Dataset split complete.")
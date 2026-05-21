import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths (relative to project root)
CSV_FILE = PROJECT_ROOT / "dataset" / "Data_Entry_2017.csv"
IMAGES_DIR = PROJECT_ROOT / "nih_images" / "images-224" / "images-224"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "chestxray14"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"

# Create output directories
for path in [TRAIN_DIR / "NORMAL", TRAIN_DIR / "PNEUMONIA", 
             VAL_DIR / "NORMAL", VAL_DIR / "PNEUMONIA"]:
    path.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ChestX-ray14 Dataset Preparation Script")
print("=" * 70)

# Step 1: Validate input files
print("\n[1/5] Validating input files...")

if not CSV_FILE.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
print(f"✓ Found CSV: {CSV_FILE}")

if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")
print(f"✓ Found images directory: {IMAGES_DIR}")

# Step 2: Read and filter CSV
print("\n[2/5] Reading and filtering CSV...")

try:
    df = pd.read_csv(CSV_FILE)
except Exception as e:
    raise Exception(f"Error reading CSV: {e}")

print(f"Total rows in CSV: {len(df)}")

# Filter for NORMAL and PNEUMONIA only
df_normal = df[df["Finding Labels"] == "No Finding"].copy()
df_pneumonia = df[df["Finding Labels"].str.contains("Pneumonia", case=False, na=False)].copy()

print(f"NORMAL images (No Finding): {len(df_normal)}")
print(f"PNEUMONIA images (contains 'Pneumonia'): {len(df_pneumonia)}")

# Combine and label
df_normal["class"] = "NORMAL"
df_pneumonia["class"] = "PNEUMONIA"
df_filtered = pd.concat([df_normal, df_pneumonia], ignore_index=True)

print(f"Total images to process: {len(df_filtered)}")

# Step 3: Stratified train/val split (80/20)
print("\n[3/5] Creating stratified train/validation split (80/20)...")

train_df, val_df = train_test_split(
    df_filtered,
    test_size=0.2,
    random_state=42,
    stratify=df_filtered["class"]
)

print(f"Training set: {len(train_df)} images")
print(f"Validation set: {len(val_df)} images")

# Step 4: Copy images
print("\n[4/5] Copying images to destination folders...")

def copy_images(df, split_name, split_dir):
    """Copy images from source to destination based on class."""
    skipped = 0
    copied = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        image_name = row["Image Index"]
        class_label = row["class"]
        
        src_path = IMAGES_DIR / image_name
        dst_dir = split_dir / class_label
        dst_path = dst_dir / image_name
        
        if not src_path.exists():
            print(f"⚠ Skipped (not found): {image_name}")
            skipped += 1
            continue
        
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"⚠ Error copying {image_name}: {e}")
            skipped += 1
    
    return copied, skipped

# Copy training images
train_copied, train_skipped = copy_images(train_df, "Training", TRAIN_DIR)

# Copy validation images
val_copied, val_skipped = copy_images(val_df, "Validation", VAL_DIR)

# Step 5: Summary statistics
print("\n[5/5] Summary Statistics")
print("=" * 70)

# Count actual files
train_normal_count = len(list((TRAIN_DIR / "NORMAL").glob("*.png")))
train_pneumonia_count = len(list((TRAIN_DIR / "PNEUMONIA").glob("*.png")))
val_normal_count = len(list((VAL_DIR / "NORMAL").glob("*.png")))
val_pneumonia_count = len(list((VAL_DIR / "PNEUMONIA").glob("*.png")))

print(f"\n📊 Dataset Summary:")
print(f"{'─' * 70}")
print(f"Total NORMAL images:     {train_normal_count + val_normal_count}")
print(f"Total PNEUMONIA images:  {train_pneumonia_count + val_pneumonia_count}")
print(f"{'─' * 70}")
print(f"\n🏋️  Training Set:")
print(f"  NORMAL:    {train_normal_count}")
print(f"  PNEUMONIA: {train_pneumonia_count}")
print(f"  Total:     {train_normal_count + train_pneumonia_count}")

print(f"\n✅ Validation Set:")
print(f"  NORMAL:    {val_normal_count}")
print(f"  PNEUMONIA: {val_pneumonia_count}")
print(f"  Total:     {val_normal_count + val_pneumonia_count}")

print(f"\n⚠️  Summary:")
print(f"  Training copied:   {train_copied} | Skipped: {train_skipped}")
print(f"  Validation copied: {val_copied} | Skipped: {val_skipped}")

print(f"\n📁 Output directory: {OUTPUT_DIR}")
print("=" * 70)
print("✓ ChestX-ray14 dataset preparation complete!")
print("=" * 70)

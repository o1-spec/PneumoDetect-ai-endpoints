import os
import shutil
import pandas as pd
from tqdm import tqdm

# Paths
CSV_PATH = "dataset/stage2_train_metadata.csv"
IMAGES_PATH = "dataset/Images"

OUTPUT_NORMAL = "dataset/processed/NORMAL"
OUTPUT_PNEUMONIA = "dataset/processed/PNEUMONIA"

# Create folders
os.makedirs(OUTPUT_NORMAL, exist_ok=True)
os.makedirs(OUTPUT_PNEUMONIA, exist_ok=True)

# Read CSV
df = pd.read_csv(CSV_PATH)

# Remove duplicates
df = df.drop_duplicates(subset=["patientId"])

print("Processing images...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    patient_id = row["patientId"]

    # Target column
    # 1 = Pneumonia
    # 0 = Normal
    target = row["Target"]

    image_name = f"{patient_id}.png"

    source_path = os.path.join(IMAGES_PATH, image_name)

    if not os.path.exists(source_path):
        continue

    # Destination
    if target == 1:
        dest_path = os.path.join(OUTPUT_PNEUMONIA, image_name)
    else:
        dest_path = os.path.join(OUTPUT_NORMAL, image_name)

    shutil.copy(source_path, dest_path)

print("Done organizing dataset!")
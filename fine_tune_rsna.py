import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, DenseNet121, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Configuration
DATASET_PATH = "dataset/final"  # RSNA dataset
PRETRAINED_MODELS_DIR = "models/pretrained"
FINETUNED_MODELS_DIR = "models/finetuned"
LOGS_DIR = "logs/finetune"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20  # Fine-tuning usually needs fewer epochs
FINETUNE_LR = 0.00001  # Lower learning rate for fine-tuning
UNFREEZE_AT = 100  # Unfreeze layers after this layer index

# Architecture (MUST match pretrained model)
MODEL_ARCHITECTURE = "mobilenetv2"
PRETRAINED_WEIGHTS = os.path.join(
    PRETRAINED_MODELS_DIR,
    f"{MODEL_ARCHITECTURE}_chestxray14_weights.weights.h5"
)
PRETRAINED_METADATA = os.path.join(
    PRETRAINED_MODELS_DIR,
    f"{MODEL_ARCHITECTURE}_chestxray14_metadata.json"
)

# Create directories
os.makedirs(FINETUNED_MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("=" * 80)
print("RSNA Fine-tuning Script")
print("=" * 80)
print(f"\n📋 Configuration:")
print(f"  Architecture: {MODEL_ARCHITECTURE}")
print(f"  Pretrained weights: {PRETRAINED_WEIGHTS}")
print(f"  Fine-tuning LR: {FINETUNE_LR}")

# ============================================================================
# Step 1: Validate Pretrained Model
# ============================================================================
print("\n[1/6] Validating pretrained model...")

if not os.path.exists(PRETRAINED_WEIGHTS):
    raise FileNotFoundError(
        f"Pretrained weights not found: {PRETRAINED_WEIGHTS}\n"
        f"Please run: python pretrain_chestxray14.py"
    )
print(f"✓ Pretrained weights found: {PRETRAINED_WEIGHTS}")

if os.path.exists(PRETRAINED_METADATA):
    with open(PRETRAINED_METADATA, "r") as f:
        pretrain_metadata = json.load(f)
    print(f"✓ Pretrained metadata loaded")
    print(f"  - Dataset: {pretrain_metadata['dataset']}")
    print(f"  - Val AUC: {pretrain_metadata['final_metrics']['val_auc']:.4f}")
else:
    print(f"⚠ Metadata not found (continuing anyway)")
    pretrain_metadata = None

# ============================================================================
# Step 2: Load RSNA Datasets
# ============================================================================
print("\n[2/6] Loading RSNA dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "val"),
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = train_ds.class_names
print(f"✓ Classes: {class_names}")
print(f"✓ Training batches: {len(train_ds)}")
print(f"✓ Validation batches: {len(val_ds)}")

# ============================================================================
# Step 3: Data Augmentation (lighter for fine-tuning)
# ============================================================================
print("\n[3/6] Setting up data augmentation and optimization...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),  # Lighter augmentation for fine-tuning
    layers.RandomZoom(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("✓ Data augmentation (lighter for fine-tuning)")
print("✓ Prefetching enabled")

# ============================================================================
# Step 4: Build Model with Pretrained Weights
# ============================================================================
print(f"\n[4/6] Building {MODEL_ARCHITECTURE.upper()} with pretrained weights...")

# Select base model
if MODEL_ARCHITECTURE == "mobilenetv2":
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
elif MODEL_ARCHITECTURE == "densenet121":
    base_model = DenseNet121(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
elif MODEL_ARCHITECTURE == "resnet50":
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
else:
    raise ValueError(f"Unknown architecture: {MODEL_ARCHITECTURE}")

# Freeze base model initially
base_model.trainable = False
print(f"✓ Base model loaded: {MODEL_ARCHITECTURE}")

# Build full model (SAME architecture as pretrain)
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# Preprocessing
if MODEL_ARCHITECTURE == "mobilenetv2":
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
elif MODEL_ARCHITECTURE == "densenet121":
    x = tf.keras.applications.densenet.preprocess_input(x)
elif MODEL_ARCHITECTURE == "resnet50":
    x = tf.keras.applications.resnet50.preprocess_input(x)

x = base_model(x, training=False)

# SAME classifier head as pretrain
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

print("✓ Model architecture created (identical to pretrain)")

# ============================================================================
# Step 5: Load Pretrained Weights
# ============================================================================
print("\n[5/6] Loading pretrained weights...")

try:
    model.load_weights(PRETRAINED_WEIGHTS)
    print(f"✓ Pretrained weights loaded successfully")
    print(f"  File: {PRETRAINED_WEIGHTS}")
except Exception as e:
    raise Exception(f"Error loading weights: {e}")

# ============================================================================
# Step 6: Prepare for Fine-tuning
# ============================================================================
print("\n[6/6] Preparing for fine-tuning...")

# Unfreeze top layers of base model for fine-tuning
# First, make base_model trainable
base_model.trainable = True

# Then freeze the bottom layers (keep early features frozen)
for layer in base_model.layers[:UNFREEZE_AT]:
    layer.trainable = False

# Count trainable layers
trainable_count = sum(1 for layer in model.layers if layer.trainable)
frozen_count = sum(1 for layer in model.layers if not layer.trainable)

print(f"✓ Unfroze layers after index {UNFREEZE_AT}")
print(f"  Trainable layers: {trainable_count}")
print(f"  Frozen layers: {frozen_count}")

# Compile with LOWER learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print(f"✓ Model compiled:")
print(f"  Optimizer: Adam (lr={FINETUNE_LR})")
print(f"  Loss: Binary Crossentropy")

model.summary()

# ============================================================================
# Step 7: Calculate Class Weights for RSNA
# ============================================================================
print("\n[7/7] Calculating class weights for RSNA dataset...")

# Get class distribution from training data
all_labels = []
for _, labels in train_ds:
    all_labels.extend(labels.numpy().flatten())
all_labels = np.array(all_labels)

classes = np.array([0, 1])
weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=all_labels,
)

class_weights_rsna = {0: float(weights_array[0]), 1: float(weights_array[1])}
print(f"✓ RSNA class weights:")
print(f"  NORMAL (class 0): {class_weights_rsna[0]:.4f}")
print(f"  PNEUMONIA (class 1): {class_weights_rsna[1]:.4f}")

# ============================================================================
# Step 8: Callbacks
# ============================================================================
print("\n[8/8] Setting up callbacks...")

callbacks = [
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        os.path.join(FINETUNED_MODELS_DIR, f"{MODEL_ARCHITECTURE}_rsna_best.keras"),
        monitor="val_auc",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1,
    ),
]

print("✓ Callbacks configured")

# ============================================================================
# Step 9: Train (Fine-tune)
# ============================================================================
print("\n" + "=" * 80)
print("FINE-TUNING ON RSNA")
print("=" * 80)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights_rsna,
    verbose=1,
)

print("=" * 80)
print("✓ Fine-tuning complete!")

# ============================================================================
# Step 10: Save Final Model & Metadata
# ============================================================================
print("\n[9/9] Saving fine-tuned model and metadata...")

model_path = os.path.join(
    FINETUNED_MODELS_DIR,
    f"{MODEL_ARCHITECTURE}_rsna_final.keras"
)
model.save(model_path)
print(f"✓ Final model saved: {model_path}")

weights_path = os.path.join(
    FINETUNED_MODELS_DIR,
    f"{MODEL_ARCHITECTURE}_rsna_weights.weights.h5"
)
model.save_weights(weights_path)
print(f"✓ Weights saved: {weights_path}")

# Save metadata
metadata = {
    "architecture": MODEL_ARCHITECTURE,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "finetune_lr": FINETUNE_LR,
    "pretrained_from": "ChestX-ray14",
    "pretrained_weights": PRETRAINED_WEIGHTS,
    "class_weights": class_weights_rsna,
    "final_metrics": {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "train_loss": float(history.history["loss"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "val_auc": float(history.history["val_auc"][-1]),
        "val_precision": float(history.history["val_precision"][-1]),
        "val_recall": float(history.history["val_recall"][-1]),
    },
}

metadata_path = os.path.join(
    FINETUNED_MODELS_DIR,
    f"{MODEL_ARCHITECTURE}_rsna_metadata.json"
)
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

import shutil

# Copy the best fine-tuned model to the main model path for evaluate.py and the Flask API
best_model_src = os.path.join(FINETUNED_MODELS_DIR, f"{MODEL_ARCHITECTURE}_rsna_best.keras")
main_model_dst = "models/pneumonia_model.keras"
if os.path.exists(best_model_src):
    # Backup existing baseline if it hasn't been backed up already
    baseline_backup = "models/pneumonia_model_baseline.keras"
    if os.path.exists(main_model_dst) and not os.path.exists(baseline_backup):
        shutil.copy2(main_model_dst, baseline_backup)
        print(f"✓ Backed up original baseline model to: {baseline_backup}")
    
    shutil.copy2(best_model_src, main_model_dst)
    print(f"✓ Copied best fine-tuned model to: {main_model_dst}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("FINE-TUNING SUMMARY")
print("=" * 80)
print(f"\n📊 Final Metrics:")
print(f"  Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Validation AUC:      {history.history['val_auc'][-1]:.4f}")
print(f"  Validation Precision: {history.history['val_precision'][-1]:.4f}")
print(f"  Validation Recall:   {history.history['val_recall'][-1]:.4f}")

print(f"\n🔄 Transfer Learning Pipeline:")
print(f"  1. ImageNet (pre-trained base models)")
print(f"  2. ChestX-ray14 (domain pretraining)")
print(f"  3. RSNA (task-specific fine-tuning) ✓ COMPLETE")

print(f"\n💾 Saved Artifacts:")
print(f"  1. Final model:       {model_path}")
print(f"  2. Model weights:     {weights_path}")
print(f"  3. Metadata:          {metadata_path}")

print(f"\n📝 Next Steps:")
print(f"  1. Evaluate on test set: python evaluate.py")
print(f"  2. Deploy to Flask API: python app.py")
print(f"  3. Export for production (ONNX/TFLite)")

print("\n" + "=" * 80)

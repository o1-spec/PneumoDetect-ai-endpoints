import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Configuration
DATASET_PATH = "dataset/final"  # RSNA dataset
PRETRAINED_MODELS_DIR = "models/pretrained"
FINETUNED_MODELS_DIR = "models/finetuned"
LOGS_DIR = "logs/finetune"

# ============================================================================
# ARCHITECTURE CONFIGURATION (MUST MATCH PRETRAINING)
# ============================================================================
# Toggle between "resnet50" and "densenet121"
MODEL_ARCHITECTURE = "resnet50"  # Options: resnet50, densenet121

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
FINETUNE_LR = 0.00001  # Lighter LR for fine-tuning

# Dynamic Paths
PRETRAINED_MODEL_DIR = os.path.join(PRETRAINED_MODELS_DIR, MODEL_ARCHITECTURE)
FINETUNED_MODEL_DIR = os.path.join(FINETUNED_MODELS_DIR, MODEL_ARCHITECTURE)

PRETRAINED_WEIGHTS = os.path.join(
    PRETRAINED_MODEL_DIR,
    f"{MODEL_ARCHITECTURE}_chestxray14_weights.weights.h5"
)
PRETRAINED_METADATA = os.path.join(
    PRETRAINED_MODEL_DIR,
    f"{MODEL_ARCHITECTURE}_chestxray14_metadata.json"
)

# Create directories
os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("=" * 80)
print(f"RSNA Pneumonia Fine-tuning: {MODEL_ARCHITECTURE.upper()}")
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
        f"Please run pretraining first: python pretrain_chestxray14.py"
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
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("✓ Data augmentation (lighter for fine-tuning)")
print("✓ Prefetching enabled")

# ============================================================================
# Step 4: Build Model
# ============================================================================
print(f"\n[4/6] Building {MODEL_ARCHITECTURE.upper()} model...")

if MODEL_ARCHITECTURE == "resnet50":
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        name="resnet50"
    )
elif MODEL_ARCHITECTURE == "densenet121":
    base_model = DenseNet121(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        name="densenet121"
    )
else:
    raise ValueError(f"Unknown architecture: {MODEL_ARCHITECTURE}")

# Base model is initially frozen
base_model.trainable = False
print(f"✓ Base model loaded: {MODEL_ARCHITECTURE}")

# Build full model (SAME architecture and head names as pretraining)
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# Preprocessing
if MODEL_ARCHITECTURE == "resnet50":
    x = tf.keras.applications.resnet50.preprocess_input(x)
elif MODEL_ARCHITECTURE == "densenet121":
    x = tf.keras.applications.densenet.preprocess_input(x)

x = base_model(x, training=False)

# Custom classifier head (with identical names to ensure perfect weight restoration)
x = layers.GlobalAveragePooling2D(name="head_gap")(x)
x = layers.Dense(256, activation="relu", name="head_dense1")(x)
x = layers.BatchNormalization(name="head_bn1")(x)
x = layers.Dropout(0.5, name="head_dropout1")(x)
x = layers.Dense(128, activation="relu", name="head_dense2")(x)
x = layers.BatchNormalization(name="head_bn2")(x)
x = layers.Dropout(0.3, name="head_dropout2")(x)
outputs = layers.Dense(1, activation="sigmoid", name="head_output")(x)

model = tf.keras.Model(inputs, outputs)
print("✓ Model structure created")

# ============================================================================
# Step 5: Load Pretrained Weights
# ============================================================================
print("\n[5/6] Loading pretrained ChestX-ray14 weights...")

try:
    model.load_weights(PRETRAINED_WEIGHTS)
    print(f"✓ Pretrained weights loaded successfully: {PRETRAINED_WEIGHTS}")
except Exception as e:
    raise RuntimeError(f"Failed to load weights: {e}")

# ============================================================================
# Step 6: Block-Name Based Unfreezing for Fine-tuning
# ============================================================================
print("\n[6/6] Configuring trainable layers...")

# First make the base model trainable
base_model.trainable = True

# Freeze everything except the final semantic block (conv5_ block)
# For both ResNet50 and DenseNet121, block 5 layers are named start with "conv5_"
for layer in base_model.layers:
    if layer.name.startswith("conv5_"):
        layer.trainable = True
    else:
        layer.trainable = False

# Count layers
total_layers = len(model.layers)
trainable_layers = sum(1 for layer in model.layers if layer.trainable)
frozen_layers = sum(1 for layer in model.layers if not layer.trainable)

print(f"✓ Safe block-name based freezing applied (unfroze 'conv5_*' blocks)")
print(f"  - Total layers in model: {total_layers}")
print(f"  - Trainable layers: {trainable_layers}")
print(f"  - Frozen layers: {frozen_layers}")

# Compile with low learning rate
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

print(f"✓ Model compiled (lr={FINETUNE_LR})")
model.summary()

# ============================================================================
# Step 7: Class Weights
# ============================================================================
print("\n[7/7] Computing class weights...")

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
print(f"✓ Class weights for RSNA dataset:")
print(f"  NORMAL (class 0): {class_weights_rsna[0]:.4f}")
print(f"  PNEUMONIA (class 1): {class_weights_rsna[1]:.4f}")

# ============================================================================
# Step 8: Callbacks
# ============================================================================
print("\n[8/8] Setting up callbacks...")

best_model_path = os.path.join(FINETUNED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_rsna_best.keras")

callbacks = [
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        best_model_path,
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
        log_dir=os.path.join(LOGS_DIR, MODEL_ARCHITECTURE),
        histogram_freq=1,
    ),
]

print(f"✓ Callbacks configured")
print(f"  Checkpoint: {best_model_path}")

# ============================================================================
# Step 9: Fine-tune model
# ============================================================================
print("\n" + "=" * 80)
print(f"FINE-TUNING {MODEL_ARCHITECTURE.upper()} ON RSNA")
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

final_model_path = os.path.join(FINETUNED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_rsna_final.keras")
model.save(final_model_path)
print(f"✓ Final model saved: {final_model_path}")

weights_path = os.path.join(FINETUNED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_rsna_weights.weights.h5")
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

metadata_path = os.path.join(FINETUNED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_rsna_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print(f"FINE-TUNING SUMMARY: {MODEL_ARCHITECTURE.upper()}")
print("=" * 80)
print(f"\n📊 Final Metrics:")
print(f"  Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Validation AUC:      {history.history['val_auc'][-1]:.4f}")
print(f"  Validation Precision: {history.history['val_precision'][-1]:.4f}")
print(f"  Validation Recall:   {history.history['val_recall'][-1]:.4f}")

print(f"\n💾 Saved Artifacts:")
print(f"  1. Best model:        {best_model_path}")
print(f"  2. Final model:       {final_model_path}")
print(f"  3. Model weights:     {weights_path}")
print(f"  4. Metadata:          {metadata_path}")
print("=" * 80)

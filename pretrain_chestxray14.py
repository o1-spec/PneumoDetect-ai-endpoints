import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path

# Configuration
DATASET_PATH = "dataset/chestxray14"
PRETRAINED_MODELS_DIR = "models/pretrained"
LOGS_DIR = "logs/pretrain"

# ============================================================================
# ARCHITECTURE CONFIGURATION
# ============================================================================
# Toggle between "resnet50" and "densenet121"
MODEL_ARCHITECTURE = "resnet50"  # Options: resnet50, densenet121

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 4  # Domain pretraining epochs
INITIAL_LR = 0.0001

# Dynamic Paths based on architecture
PRETRAINED_MODEL_DIR = os.path.join(PRETRAINED_MODELS_DIR, MODEL_ARCHITECTURE)
os.makedirs(PRETRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("=" * 80)
print(f"ChestX-ray14 Domain Pretraining: {MODEL_ARCHITECTURE.upper()}")
print("=" * 80)

# ============================================================================
# Step 1: Load Datasets
# ============================================================================
print("\n[1/6] Loading datasets...")

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
# Step 2: Data Augmentation & Optimization
# ============================================================================
print("\n[2/6] Setting up data augmentation and optimization...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("✓ Data augmentation layers added")
print("✓ Prefetching enabled")

# Calculate class weights (ChestX-ray14 has roughly 45:1 normal-to-pneumonia imbalance)
total_normal = 48330
total_pneumonia = 1082

classes = np.array([0, 1])
y_train = [0] * total_normal + [1] * total_pneumonia

weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = {0: float(weights_array[0]), 1: float(weights_array[1])}
print(f"✓ Class weights calculated (balanced):")
print(f"  NORMAL (class 0): {class_weights[0]:.4f}")
print(f"  PNEUMONIA (class 1): {class_weights[1]:.4f}")

# ============================================================================
# Step 3: Build Model
# ============================================================================
print(f"\n[3/6] Building {MODEL_ARCHITECTURE.upper()} model...")

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

# Freeze base model initially
base_model.trainable = False
print(f"✓ Base model loaded: {MODEL_ARCHITECTURE}")
print(f"✓ Base model layers frozen for initial head training")

# Build full model
inputs = layers.Input(shape=(224, 224, 3))

# Data augmentation
x = data_augmentation(inputs)

# Preprocessing
if MODEL_ARCHITECTURE == "resnet50":
    x = tf.keras.applications.resnet50.preprocess_input(x)
elif MODEL_ARCHITECTURE == "densenet121":
    x = tf.keras.applications.densenet.preprocess_input(x)

# Base model
x = base_model(x, training=False)

# Custom classifier head - Prefixed with "head_" for easy Grad-CAM parsing
x = layers.GlobalAveragePooling2D(name="head_gap")(x)
x = layers.Dense(256, activation="relu", name="head_dense1")(x)
x = layers.BatchNormalization(name="head_bn1")(x)
x = layers.Dropout(0.5, name="head_dropout1")(x)
x = layers.Dense(128, activation="relu", name="head_dense2")(x)
x = layers.BatchNormalization(name="head_bn2")(x)
x = layers.Dropout(0.3, name="head_dropout2")(x)
outputs = layers.Dense(1, activation="sigmoid", name="head_output")(x)

model = tf.keras.Model(inputs, outputs)

print("✓ Model architecture created with 'head_' layer prefixes")
print(f"  - Input: 224×224×3")
print(f"  - Base: {MODEL_ARCHITECTURE}")

# ============================================================================
# Step 4: Compile Model
# ============================================================================
print("\n[4/6] Compiling model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("✓ Optimizer: Adam (lr=0.0001)")
print("✓ Loss: Binary Crossentropy")
print("✓ Metrics: Accuracy, Precision, Recall, AUC")

# Check if a previous checkpoint exists to resume training
best_model_path = os.path.join(PRETRAINED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_chestxray14_best.keras")
if os.path.exists(best_model_path):
    print("\n" + "=" * 80)
    print(f"✓ Found existing checkpoint: {best_model_path}")
    print("Loading weights to resume pre-training...")
    try:
        model.load_weights(best_model_path)
        print("✓ Weights loaded successfully!")
    except Exception as e:
        print(f"⚠ Failed to load weights: {e}. Starting from scratch.")
    print("=" * 80 + "\n")

model.summary()

# ============================================================================
# Step 5: Callbacks
# ============================================================================
print("\n[5/6] Setting up callbacks...")

best_model_path = os.path.join(PRETRAINED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_chestxray14_best.keras")

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

print(f"✓ Early stopping: patience=5, monitor=val_auc")
print(f"✓ Model checkpoint: {best_model_path}")
print(f"✓ LR reduction: factor=0.5, patience=3")
print(f"✓ Tensorboard logging enabled under logs/pretrain/{MODEL_ARCHITECTURE}")

# ============================================================================
# Step 6: Train Model
# ============================================================================
print("\n[6/6] Training model on ChestX-ray14...")
print("=" * 80)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
)

print("=" * 80)
print("✓ Training complete!")

# ============================================================================
# Step 7: Save Training History & Metadata
# ============================================================================
print("\n[7/7] Saving metadata and weights...")

# Save final model
final_model_path = os.path.join(PRETRAINED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_chestxray14_final.keras")
model.save(final_model_path)
print(f"✓ Full final model saved: {final_model_path}")

# Save weights for transfer learning
weights_path = os.path.join(PRETRAINED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_chestxray14_weights.weights.h5")
model.save_weights(weights_path)
print(f"✓ Model weights saved: {weights_path}")

# Save base model weights only
base_model_path = os.path.join(PRETRAINED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_chestxray14_base_weights.weights.h5")
base_model.save_weights(base_model_path)
print(f"✓ Base model weights saved: {base_model_path}")

# Save metadata
metadata = {
    "architecture": MODEL_ARCHITECTURE,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "initial_lr": INITIAL_LR,
    "class_weights": {
        "NORMAL": float(class_weights[0]),
        "PNEUMONIA": float(class_weights[1]),
    },
    "dataset": "ChestX-ray14",
    "dataset_sizes": {
        "train_normal": 48330,
        "train_pneumonia": 1082,
        "val_normal": 12082,
        "val_pneumonia": 271,
    },
    "final_metrics": {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "train_loss": float(history.history["loss"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "val_auc": float(history.history["val_auc"][-1]),
    },
}

metadata_path = os.path.join(PRETRAINED_MODEL_DIR, f"{MODEL_ARCHITECTURE}_chestxray14_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("PRETRAINING SUMMARY")
print("=" * 80)
print(f"\n📊 Final Metrics:")
print(f"  Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Validation AUC:      {history.history['val_auc'][-1]:.4f}")
print(f"  Validation Loss:     {history.history['val_loss'][-1]:.4f}")

print(f"\n💾 Saved Artifacts:")
print(f"  1. Best model:        {best_model_path}")
print(f"  2. Final model:       {final_model_path}")
print(f"  3. Model weights:     {weights_path}")
print(f"  4. Metadata:          {metadata_path}")

print(f"\n📝 Next Steps:")
print(f"  1. Review metrics in tensorboard:")
print(f"     tensorboard --logdir={LOGS_DIR}/{MODEL_ARCHITECTURE}")
print(f"  2. Fine-tune on RSNA dataset by switching configuration in fine_tune_rsna.py")
print("=" * 80)

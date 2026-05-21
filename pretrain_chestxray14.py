import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, DenseNet121, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import json
from pathlib import Path

# Configuration
DATASET_PATH = "dataset/chestxray14"
PRETRAINED_MODELS_DIR = "models/pretrained"
LOGS_DIR = "logs/pretrain"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Start with 10 for domain adaptation. Increase if needed after evaluation
INITIAL_LR = 0.0001
MODEL_ARCHITECTURE = "mobilenetv2"  # Options: mobilenetv2, densenet121, resnet50

# Create directories
os.makedirs(PRETRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("=" * 80)
print("ChestX-ray14 Pretraining Script")
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

# Data augmentation for training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("✓ Data augmentation layers added")
print("✓ Prefetching enabled")

# Calculate class weights (handle severe imbalance)
# Training set: 48,330 NORMAL, 1,082 PNEUMONIA
total_normal = 48330
total_pneumonia = 1082

# Use sklearn's balanced class weight computation
classes = np.array([0, 1])
y_train = [0] * total_normal + [1] * total_pneumonia

weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = {0: float(weights_array[0]), 1: float(weights_array[1])}
print(f"✓ Class weights calculated (sklearn.utils.class_weight):")
print(f"  NORMAL (class 0): {class_weights[0]:.4f}")
print(f"  PNEUMONIA (class 1): {class_weights[1]:.4f}")

# ============================================================================
# Step 3: Build Model
# ============================================================================
print(f"\n[3/6] Building {MODEL_ARCHITECTURE.upper()} model...")

# Select base model
if MODEL_ARCHITECTURE == "mobilenetv2":
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    model_name = "mobilenetv2_chestxray14"
elif MODEL_ARCHITECTURE == "densenet121":
    base_model = DenseNet121(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    model_name = "densenet121_chestxray14"
elif MODEL_ARCHITECTURE == "resnet50":
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    model_name = "resnet50_chestxray14"
else:
    raise ValueError(f"Unknown architecture: {MODEL_ARCHITECTURE}")

# Freeze base model initially
base_model.trainable = False
print(f"✓ Base model loaded: {MODEL_ARCHITECTURE}")
print(f"✓ Base model layers frozen for initial training")

# Build full model
inputs = layers.Input(shape=(224, 224, 3))

# Data augmentation
x = data_augmentation(inputs)

# Preprocessing (model-specific)
if MODEL_ARCHITECTURE == "mobilenetv2":
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
elif MODEL_ARCHITECTURE == "densenet121":
    x = tf.keras.applications.densenet.preprocess_input(x)
elif MODEL_ARCHITECTURE == "resnet50":
    x = tf.keras.applications.resnet50.preprocess_input(x)

# Base model
x = base_model(x, training=False)

# Custom head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

print("✓ Model architecture created")
print(f"  - Input: 224×224×3")
print(f"  - Base: {MODEL_ARCHITECTURE}")
print(f"  - Head: GlobalAvgPool → Dense(256) → BatchNorm → Dropout(0.5)")
print(f"         → Dense(128) → BatchNorm → Dropout(0.3) → Dense(1, sigmoid)")

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

model.summary()

# ============================================================================
# Step 5: Callbacks
# ============================================================================
print("\n[5/6] Setting up callbacks...")

callbacks = [
    # Early stopping
    EarlyStopping(
        monitor="val_auc",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    # Model checkpoint
    ModelCheckpoint(
        os.path.join(PRETRAINED_MODELS_DIR, f"{model_name}_best.keras"),
        monitor="val_auc",
        save_best_only=True,
        verbose=1,
    ),
    # Learning rate reduction
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
    # Tensorboard logging
    tf.keras.callbacks.TensorBoard(
        log_dir=LOGS_DIR,
        histogram_freq=1,
    ),
]

print("✓ Early stopping: patience=5, monitor=val_auc")
print("✓ Model checkpoint: save best weights")
print("✓ LR reduction: factor=0.5, patience=3")
print("✓ Tensorboard logging enabled")

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

# Save full model
model_path = os.path.join(PRETRAINED_MODELS_DIR, f"{model_name}_final.keras")
model.save(model_path)
print(f"✓ Full model saved: {model_path}")

# Save just the weights for transfer learning
weights_path = os.path.join(PRETRAINED_MODELS_DIR, f"{model_name}_weights.weights.h5")
model.save_weights(weights_path)
print(f"✓ Weights saved: {weights_path}")

# Save base model weights
base_model_path = os.path.join(PRETRAINED_MODELS_DIR, f"{model_name}_base_weights.weights.h5")
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

metadata_path = os.path.join(PRETRAINED_MODELS_DIR, f"{model_name}_metadata.json")
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
print(f"  1. Full model:        {model_path}")
print(f"  2. Model weights:     {weights_path}")
print(f"  3. Base weights:      {base_model_path}")
print(f"  4. Metadata:          {metadata_path}")

print(f"\n📝 Next Steps:")
print(f"  1. Review metrics in tensorboard:")
print(f"     tensorboard --logdir={LOGS_DIR}")
print(f"  2. Use pretrained weights for fine-tuning on RSNA:")
print(f"     pretrain_path = '{weights_path}'")
print(f"     model.load_weights(pretrain_path)")
print(f"  3. Create fine_tune_rsna.py script")

print("\n" + "=" * 80)

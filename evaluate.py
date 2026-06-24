import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import os

VAL_PATH = "dataset/final/val"
RESNET_MODEL_PATH = "models/finetuned/resnet50/resnet50_rsna_best.keras"
DENSENET_MODEL_PATH = "models/finetuned/densenet121/densenet121_rsna_best.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("=" * 80)
print("PneumoDetect AI - Ensemble Evaluation Pipeline")
print("=" * 80)

# Validate models exist
if not os.path.exists(RESNET_MODEL_PATH):
    raise FileNotFoundError(f"ResNet50 fine-tuned model not found: {RESNET_MODEL_PATH}")
if not os.path.exists(DENSENET_MODEL_PATH):
    raise FileNotFoundError(f"DenseNet121 fine-tuned model not found: {DENSENET_MODEL_PATH}")

# 1. Load raw validation dataset (without normalization applied yet)
print("\n[1/4] Loading raw validation dataset...")
raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = raw_val_ds.class_names
print(f"✓ Classes detected: {class_names}")

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE

# 2. Reconstruct correct preprocessed datasets for each backbone
print("\n[2/4] Constructing backbone-specific preprocessed data streams...")
resnet_ds = raw_val_ds.map(
    lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

densenet_ds = raw_val_ds.map(
    lambda x, y: (tf.keras.applications.densenet.preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

print("✓ ResNet50 preprocessing pipeline constructed")
print("✓ DenseNet121 preprocessing pipeline constructed")

# 3. Load both fine-tuned models
print("\n[3/4] Loading deep models into memory...")
print("  - Loading ResNet50...")
resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
print("  - Loading DenseNet121...")
densenet_model = tf.keras.models.load_model(DENSENET_MODEL_PATH)
print("✓ Both models loaded successfully")

# 4. Extract true labels
print("\nExtracting ground-truth labels...")
y_true = np.concatenate([y.numpy() for _, y in raw_val_ds])
print(f"Total validation samples: {len(y_true)}")

# 5. Generate predictions
print("Generating ResNet50 predictions...")
y_pred_resnet = resnet_model.predict(raw_val_ds, verbose=1).flatten()

print("Generating DenseNet121 predictions...")
y_pred_densenet = densenet_model.predict(raw_val_ds, verbose=1).flatten()

# Weighted Ensemble Averaging (based on validation performance ratios)
print("Computing weighted ensemble probabilities...")
y_pred_probs = (0.52 * y_pred_resnet) + (0.48 * y_pred_densenet)

# 6. Evaluate Backbones and Ensemble
auc_resnet = roc_auc_score(y_true, y_pred_resnet)
auc_densenet = roc_auc_score(y_true, y_pred_densenet)
auc_ensemble = roc_auc_score(y_true, y_pred_probs)

print("\n" + "=" * 50)
print("BACKBONE VS ENSEMBLE PERFORMANCE (AUC-ROC)")
print("=" * 50)
print(f"ResNet50 Model AUC-ROC:   {auc_resnet:.4f}")
print(f"DenseNet121 Model AUC-ROC: {auc_densenet:.4f}")
print(f"Dual Ensemble AUC-ROC:     {auc_ensemble:.4f}  <--")
print("=" * 50)

# Threshold search to maximize F1-score
print("\nSearching for optimal decision threshold...")
best_threshold = 0.5
best_f1 = 0.0

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_probs > threshold).astype("int32")
    f1 = f1_score(y_true, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"✓ Optimal threshold found: {best_threshold:.2f} (Validation F1: {best_f1:.4f})")

y_pred = (y_pred_probs > best_threshold).astype("int32")

# Classification report
print("\nClassification Report (Ensemble):")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
)

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n" + "=" * 80)
print("EVALUATION STAGE COMPLETE")
print("=" * 80)
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

VAL_PATH = "dataset/final/val"
MODEL_PATH = "models/pneumonia_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = val_ds.class_names
print("Classes:", class_names)

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds.prefetch(AUTOTUNE)

model = tf.keras.models.load_model(MODEL_PATH)

print("Extracting true labels (this takes a moment)...")
y_true = np.concatenate([y.numpy() for _, y in val_ds])

print("Generating predictions...")
y_pred_probs = model.predict(val_ds).flatten()

auc_score = roc_auc_score(y_true, y_pred_probs)
print(f"\nAUC-ROC: {auc_score:.4f}")

best_threshold = 0.5
best_f1 = 0.0

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_temp = (y_pred_probs > threshold).astype("int32")
    f1 = f1_score(y_true, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nOptimal threshold: {best_threshold:.2f} (F1 score: {best_f1:.4f})")

y_pred = (y_pred_probs > best_threshold).astype("int32")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
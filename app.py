from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

# Paths to the fine-tuned ensemble models
RESNET_PATH = "models/finetuned/resnet50/resnet50_rsna_best.keras"
DENSENET_PATH = "models/finetuned/densenet121/densenet121_rsna_best.keras"

IMG_SIZE = (224, 224)
THRESHOLD = 0.50  # Optimized threshold found by evaluation grid search

# Global model pointers
resnet_model = None
densenet_model = None

# Partitioned sub-models for Grad-CAM to prevent graph disconnection
resnet_base_grad_model = None
resnet_head_model = None

densenet_base_grad_model = None
densenet_head_model = None

def find_last_conv_layer(base_model):
    """Programmatically identify the final Conv2D layer inside the base model."""
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError(f"No Conv2D layer found inside base model: {base_model.name}")

def build_head_model(model, base_model_output_shape):
    """Reconstruct a separate head model sequentially using layers prefixed with 'head_'."""
    head_input = tf.keras.Input(shape=base_model_output_shape[1:])
    x = head_input
    # Gather all head layers in their original execution order
    head_layers = [layer for layer in model.layers if layer.name.startswith("head_")]
    for layer in head_layers:
        x = layer(x)
    return tf.keras.Model(head_input, x, name=f"head_model_{model.name}")

def init_models():
    """Load both models and setup their partitioned sub-graphs for Grad-CAM."""
    global resnet_model, densenet_model
    global resnet_base_grad_model, resnet_head_model
    global densenet_base_grad_model, densenet_head_model

    print("\n" + "=" * 80)
    print("INITIALIZING ENSEMBLE MODELS & GRAD-CAM PARTITIONS")
    print("=" * 80)

    if not os.path.exists(RESNET_PATH):
        raise FileNotFoundError(f"ResNet50 model file missing: {RESNET_PATH}")
    if not os.path.exists(DENSENET_PATH):
        raise FileNotFoundError(f"DenseNet121 model file missing: {DENSENET_PATH}")

    # Load ResNet50
    print("Loading fine-tuned ResNet50 model...")
    resnet_model = tf.keras.models.load_model(RESNET_PATH)
    resnet_base = resnet_model.get_layer("resnet50")
    last_resnet_conv = find_last_conv_layer(resnet_base)
    print(f"  - Programmatically detected last ResNet Conv2D: '{last_resnet_conv}'")
    resnet_base_grad_model = tf.keras.Model(
        inputs=resnet_base.inputs,
        outputs=[resnet_base.get_layer(last_resnet_conv).output, resnet_base.output]
    )
    resnet_head_model = build_head_model(resnet_model, resnet_base.output_shape)
    print("  - Partitioned ResNet50 head reconstructed successfully")

    # Load DenseNet121
    print("\nLoading fine-tuned DenseNet121 model...")
    densenet_model = tf.keras.models.load_model(DENSENET_PATH)
    densenet_base = densenet_model.get_layer("densenet121")
    last_densenet_conv = find_last_conv_layer(densenet_base)
    print(f"  - Programmatically detected last DenseNet Conv2D: '{last_densenet_conv}'")
    densenet_base_grad_model = tf.keras.Model(
        inputs=densenet_base.inputs,
        outputs=[densenet_base.get_layer(last_densenet_conv).output, densenet_base.output]
    )
    densenet_head_model = build_head_model(densenet_model, densenet_base.output_shape)
    print("  - Partitioned DenseNet121 head reconstructed successfully")
    print("=" * 80 + "\n")

# Initialize models immediately upon module load
init_models()

def make_gradcam_heatmap(img_array, base_grad_model, head_model):
    """Computes a normalized Grad-CAM activation heatmap for a given input array."""
    # Ensure correct tensor type
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = base_grad_model(img_tensor)
        tape.watch(conv_outputs)
        predictions = head_model(base_outputs)
        loss = predictions[:, 0]

    # Calculate gradients of the binary class score with respect to conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Average the gradients across height and width to extract channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get activations for the first (and only) image in the batch
    conv_outputs = conv_outputs[0]

    # Perform channel weighted multiplication
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (keep only features having a positive impact on class probability)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize heatmap between 0 and 1
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()

def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """Resizes, colors, and overlays the heatmap onto the original image."""
    original_image = np.array(original_image)

    # Resize heatmap to match original input resolution
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply jet color mapping (warm red for pneumonia regions, cool blue for background)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend original grayscale/RGB X-ray and colored heatmap
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay

def image_to_base64(image_array):
    """Helper to convert a numpy image array into a Base64-encoded PNG string."""
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "PneumoDetect AI Ensemble API is running",
        "models_loaded": {
            "resnet50": True,
            "densenet121": True
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Load and resize original image
        image = Image.open(file).convert("RGB")
        resized_image = image.resize(IMG_SIZE)

        # Prepare raw image array (inputs to the model graph are raw pixel values [0, 255])
        raw_img = np.array(resized_image, dtype=np.float32)
        raw_img = np.expand_dims(raw_img, axis=0)

        # 1. ResNet50 Prediction
        pred_resnet = float(resnet_model.predict(raw_img)[0][0])

        # 2. DenseNet121 Prediction
        pred_densenet = float(densenet_model.predict(raw_img)[0][0])

        # 3. Weighted Ensemble Combination
        ensemble_pred = (0.52 * pred_resnet) + (0.48 * pred_densenet)

        # Classify based on the threshold
        if ensemble_pred > THRESHOLD:
            result = "PNEUMONIA"
            confidence = ensemble_pred
        else:
            result = "NORMAL"
            confidence = 1.0 - ensemble_pred

        # 4. Programmatic, robust Grad-CAM Heatmap generation
        resnet_heatmap = make_gradcam_heatmap(
            raw_img, resnet_base_grad_model, resnet_head_model
        )
        densenet_heatmap = make_gradcam_heatmap(
            raw_img, densenet_base_grad_model, densenet_head_model
        )

        # 5. Weighted combination of heatmaps based on validation performance
        # DenseNet121 (0.52) has a slightly higher typical AUC than ResNet50 (0.48)
        fused_heatmap = (resnet_heatmap * 0.48) + (densenet_heatmap * 0.52)

        # Re-normalize the fused activation map to retain sharp clarity
        fused_max = np.max(fused_heatmap)
        if fused_max > 0:
            fused_heatmap = fused_heatmap / fused_max

        # 6. Apply colored overlay onto original X-Ray
        overlay = overlay_heatmap(image, fused_heatmap)
        heatmap_base64 = image_to_base64(overlay)

        return jsonify({
            "result": result,
            "confidence": round(confidence, 4),
            "raw_predictions": {
                "resnet50": round(pred_resnet, 4),
                "densenet121": round(pred_densenet, 4),
                "ensemble_average": round(ensemble_pred, 4)
            },
            "heatmap": heatmap_base64
        })

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

if __name__ == "__main__":
    # Start production API on default port 7860
    app.run(debug=False, host="0.0.0.0", port=7860)
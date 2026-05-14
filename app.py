from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("models/pneumonia_model.keras")

IMG_SIZE = (224, 224)
THRESHOLD = 0.50

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("mobilenetv2_1.00_224").get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_heatmap(original_image, heatmap, alpha=0.4):
    original_image = np.array(original_image)

    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay

def image_to_base64(image_array):
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "PneumoDetect AI API running"
    })

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    image = Image.open(file).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    prediction = model.predict(image_array)[0][0]

    if prediction > THRESHOLD:
        result = "PNEUMONIA"
        confidence = float(prediction)
    else:
        result = "NORMAL"
        confidence = float(1 - prediction)

    heatmap = make_gradcam_heatmap(image_array, model)
    overlay = overlay_heatmap(image, heatmap)
    heatmap_base64 = image_to_base64(overlay)

    return jsonify({
        "result": result,
        "confidence": round(confidence, 4),
        "raw_prediction": round(float(prediction), 4),
        "heatmap": heatmap_base64
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("models/pneumonia_model.keras")

IMG_SIZE = (224, 224)
THRESHOLD = 0.55

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

    return jsonify({
        "result": result,
        "confidence": round(confidence, 4),
        "raw_prediction": round(float(prediction), 4)
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)
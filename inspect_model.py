import tensorflow as tf
model = tf.keras.models.load_model("models/pneumonia_model.keras")
print("Model Layers:")
for layer in model.layers:
    print(layer.name)
    if isinstance(layer, tf.keras.Model):
        print(f"  Inner layers for {layer.name}:")
        for inner_layer in layer.layers[-5:]: # print last 5 inner layers
            print(f"    {inner_layer.name}")

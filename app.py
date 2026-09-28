from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the trained model from the leaf project folder
model = tf.keras.models.load_model(
    "citrus_leaf_disease_model.keras"
)

# Disease classes used in the Leaf model
class_names = [
    "Healthy",
    "Canker",
    "Black Spot",
    "Greening"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Check whether an image was uploaded
    if "file" not in request.files:
        return "No image uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No image selected"

    # Read uploaded leaf image
    image = Image.open(file).convert("RGB")

    # Your CNN was trained with 128 × 128 images
    image = image.resize((128, 128))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Run the trained Leaf CNN
    prediction = model.predict(image_array)

    # Find predicted disease
    predicted_class = np.argmax(prediction)

    # Calculate confidence
    confidence = np.max(prediction) * 100

    disease = class_names[predicted_class]

    # Display result on webpage
    return render_template(
        "index.html",
        prediction=disease,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
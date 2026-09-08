from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("model/classifier.keras")

# Load class names
with open("model/classes.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

IMAGE_SIZE = (180, 180)


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_class, confidence


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please select an image."

        else:

            file = request.files["image"]

            if file.filename == "":
                error = "Please select an image."

            else:

                filename = secure_filename(file.filename)

                filepath = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    filename
                )

                file.save(filepath)

                prediction, confidence = predict_image(filepath)

                image_url = "/" + filepath.replace("\\", "/")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
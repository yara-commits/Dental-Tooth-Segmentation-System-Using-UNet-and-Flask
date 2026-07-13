"""
Dental Tooth Segmentation Web Application
Using a trained U-Net model
Course: Introduction to Clinical Context for AI in Healthcare
"""
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# ============================
# Folder Configuration
# ============================

UPLOAD_FOLDER = "uploads"
PREDICTION_FOLDER = "predictions"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PREDICTION_FOLDER"] = PREDICTION_FOLDER

# ============================
# Load U-Net Model
# ============================

model = tf.keras.models.load_model(
    "best_unet.keras",
    compile=False
)

# ============================
# Home Page
# ============================

@app.route("/")
def index():
    return render_template("index.html")


# ============================
# Prediction
# ============================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
       return render_template("index.html")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html")

upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

file.save(upload_path)

    # -----------------------
    # Read image
    # -----------------------

image = Image.open(upload_path).convert("RGB")
image = image.resize((256, 256))

img = np.array(image, dtype=np.float32) / 255.0

input_image = np.expand_dims(img, axis=0)

    # -----------------------
    # Predict
    # -----------------------

prediction = model.predict(input_image, verbose=0)[0]

mask = (prediction > 0.5).astype(np.uint8)

mask = mask.squeeze()

mask = mask * 255

filename = os.path.basename(file.filename)
prediction_name = "prediction_" + filename

prediction_path = os.path.join(
        app.config["PREDICTION_FOLDER"],
        prediction_name
    )

cv2.imwrite(prediction_path, mask)

   return render_template(
        "index.html",
        original=upload_path,
        prediction=prediction_path
    )


# ============================
# Run App
# ============================

if __name__ == "__main__":
    app.run(debug=True)
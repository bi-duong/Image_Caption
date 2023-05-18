from flask import Flask, render_template, request
import pickle
import pyttsx3
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.utils import pad_sequences
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import pyttsx3
from PIL import Image
import time

from ImageCaption.caption import preprocessImage, GenerateSpeech

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1




@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("/index.html")


@app.route("/after", methods=["POST", "GET"])
def after():
    file = request.files["img"]
    im = file.save("static/img.jpg")
    imggg = Image.open(file)
    imggg = imggg.resize((224, 224))
    cptn = preprocessImage(imggg)
    GenerateSpeech(cptn)
    return render_template("/predict.html", data=cptn)


if (__name__) == "__main__":
    app.run(debug=False, host="0.0.0.0")


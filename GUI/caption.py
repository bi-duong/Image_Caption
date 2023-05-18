import pickle
import pyttsx3
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.utils import pad_sequences
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import pyttsx3
f = open("tokenizer.pkl", "rb")
tokenizer = pickle.load(f)

model = keras.models.load_model("model10epoch.h5")
vgg = keras.applications.vgg16.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
vgg = keras.Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

vocab_size = len(tokenizer.word_index) + 1


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    return in_text


def generate_caption(y_pred):
    print(y_pred)
    y_pred = y_pred.replace("startseq", "")
    y_pred = y_pred.replace(" endseq", ".")
    y_pred = y_pred.capitalize()
    print(y_pred)
    # GenerateSpeech(y_pred)
    return y_pred


def preprocessImage(image):

    image = img_to_array(image)
    imagek = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    imagek = preprocess_input(imagek)
    feature = vgg.predict(imagek, verbose=0)
    text = predict_caption(model, feature, tokenizer, 35)
    return generate_caption(text)


def GenerateSpeech(text):
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", rate - 90)
    engine.say(text)
    engine.runAndWait()
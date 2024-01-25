from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import cv2
import numpy as np
from skimage.transform import resize as resize_img

app = Flask(__name__)
CORS(app)

N_CLASSES = 2
IMG_SHAPE = (224, 224)

MEAN_COLOR = torch.tensor([0.4516, 0.4224, 0.4670])
STD_COLOR = torch.tensor([0.2415, 0.2177, 0.2393])

MEAN_NO_COLOR = torch.tensor(0.4603)
STD_NO_COLOR = torch.tensor(0.2265)

CLASSES_NAMES = ["HIPPO", "RHINO"]

models = {
    "vit_c": None,
    "vit_wc": None,
    "cnn_c": None,
    "cnn_wc": None,
    "resnet_c": torch.load("../models/model_resnet18_c.pch"),
    "resnet_wc": torch.load("../models/model_resnet18_wc.pch"),
}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files["image"]
        model_name = request.form["model_name"]
        with_color = request.form["with_color"] == "true"
    except Error:
        return jsonify({"prediction": None, "precision": None})

    if with_color:
        color_str = "_c"
    else:
        color_str = "_wc"

    model_name = f"{model_name}{color_str}"
    print(with_color, model_name)
    model = models[model_name]

    model.eval()

    img = np.array(Image.open(image.stream))

    if not with_color and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = resize_img(img, IMG_SHAPE, anti_aliasing=True)

    if with_color:
        H, W, C = img.shape
        mean = MEAN_COLOR
        std = STD_COLOR

    else:
        H, W = img.shape
        C = 1
        mean = MEAN_NO_COLOR
        std = STD_NO_COLOR

    img = img.reshape((1, C, W, H))

    if model_name == "resnet_wc" or model_name == "vit_wc":
        img = np.repeat(img, 3, axis=1)

    img = torch.tensor(img, dtype=torch.float32)

    normalizer = transforms.Normalize(mean, std)

    img = normalizer(img)

    with torch.no_grad():
        yhat = model(img)

    if model_name == "vit_c" or model_name == "vit_wc":
        yhat = yhat.logits

    proba = F.softmax(yhat, dim=-1)

    prediction = CLASSES_NAMES[proba.argmax()]
    precision = np.round(proba.max().item() * 100, 2)

    return jsonify({"prediction": prediction, "precision": precision})


if __name__ == "__main__":
    app.run(debug=True)

import io
import json
import os
import flask
import torch

from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request, redirect, flash, url_for
from werkzeug.utils import secure_filename


classes = {
    0: 'dogs',
    1: 'cats',
    2: 'panda'
}

app = Flask(__name__, template_folder='templates')

model = torch.jit.load("model_scripted_resnet_1.pt", map_location='cpu')
model.eval()

def transform_image(image):
    transform_image = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image))
    return transform_image(image).unsqueeze(0)

def get_prediction(image):
    tensor = transform_image(image)
    outputs = model.forward(tensor)
    print(outputs)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    if predicted_idx == 0:
        return("The predicted animal is a dog")
    elif predicted_idx == 1:
        return("The predicted animal is a cat")
    elif predicted_idx == 2:
        return("The predicted animal is a panda")
    return None

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method != "POST":
        return "Get is not supported"
    image = request.files['file']
    # image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
    img_bytes = image.read()
    class_name = get_prediction(img_bytes)

    return flask.render_template('index.html', class_name=class_name)


if __name__ == '__main__':
    app.run(debug=True)

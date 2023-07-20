import flask
import io
import string
import time
import os
import webbrowser
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, Model
import tensorflow_addons as tfa
from PIL import Image
from flask import Flask, jsonify, request, render_template
import json

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

model = tf.keras.models.load_model('/Users/ravigadgil/Downloads/UWECREUFiles/FlaskDemo/ViT-Merced/ViT-UCMerced50_50.h5', custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder})

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def predict_result(img):
    sample_to_predict = np.array(img)
    predictions = model.predict(sample_to_predict)
    classes = np.argmax(predictions, axis = 1)
    return classes[0]


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    class_json_str = json.dumps({'class': int(predict_result(img))})

    return jsonify(prediction=class_json_str)
    

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
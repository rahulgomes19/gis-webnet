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
import base64
import imghdr

app = Flask(__name__)

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

def convert_to_jpg(img):
    return img.convert("RGB")

@app.route('/', endpoint='func1')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'], endpoint='func2')
def upload_file():
   if request.method == 'POST':
    classes = ['Agricultural', 'Airplane', 'Baseball Diamond', 'Beach', 'Building', 'Chaparral', 'Dense Residential', 'Forest', 'Freeway', 'Golf Course', 'Harbor',
                'Intersection', 'Medium Residential', 'Mobile Home Park', 'Overpass', 'Parking Lot', 'River', 'Runway', 'Sparse Residential', 'Storage Tanks',
                'Tennis Court']

    img = Image.open(request.files['file'].stream)
    img_to_display = convert_to_jpg(img)

    data = io.BytesIO()
    img_to_display.save(data, "JPEG")
    
    encoded_img_data = base64.b64encode(data.getvalue())

    img_array = img.resize((256, 256))
    img_array = np.array(img_array)
    img_array = np.expand_dims(img_array, 0)

    model = tf.keras.models.load_model('/Users/ravigadgil/Downloads/UWECREUFiles/FlaskDemo/ViT-Merced/ViT-UCMerced50_50.h5', custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder})

    sample_to_predict = np.array(img_array)
    predictions = model.predict(sample_to_predict)
    class_index = np.argmax(predictions, axis = 1)

    return render_template('index.html', class_name=classes[class_index[0]], img_data=encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0')
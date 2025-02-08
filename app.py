from flask import Flask, request, jsonify, render_template

import os
import subprocess

from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')  # Ensure this serves the latest version

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)

    if result != 0:
        return jsonify({"error": "Training failed"}), 500
    return "Training done successfully!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    if 'error' in result:
        return jsonify(result), 500
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #local host

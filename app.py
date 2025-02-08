from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage  # Importing the decodeImage function
from cnnClassifier.pipeline.predict import PredictionPipeline
import os

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

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    image = request.json['image']
    
    try:
        # Decode image and save it
        decodeImage(image, clApp.filename)
        
        # Run prediction on the decoded image
        result = clApp.classifier.predict()
        
        if 'error' in result:
            return jsonify(result), 500
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)  # Localhost

from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline

# Set up environment variables (ensure proper encoding)
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Initialize the ClientApp instance globally so it can be accessed in all routes
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# Create the client app instance globally
clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()

@cross_origin()
def trainRoute():
    try:
        # If you're using DVC for model training:
        os.system("dvc repro")  # Repro is used for running DVC pipelines
        # If you're using a separate script for training:
        # os.system("python main.py")
        return jsonify({"message": "Training done successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        # Get the base64 image string from the request
        image = request.json['image']
        print("Received Image:", image[:100])  # Debug: Print the first 100 chars of the base64 string

        # Decode the base64 image and save it to the file path
        decodeImage(image, clApp.filename)

        # Run prediction
        result = clApp.classifier.predict()

        # Debugging the result
        print("Prediction Result:", result)

        # Return the result as JSON
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error: {str(e)}")  # Log the error

        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run the app (port 80 for production/Azure, port 8080 for testing)
    app.run(host='0.0.0.0', port=8080)  # Change port to 80 for production (Azure)

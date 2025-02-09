"""
Chicken Disease Detection - Flask Web Application

This module serves as the main entry point for the Chicken Disease Detection web application.
It provides a REST API for disease prediction and a web interface for users to interact with
the model. The application handles image uploads, makes predictions using a trained CNN model,
and displays training history and model performance metrics.

Main Components:
- REST API endpoints for prediction
- Web interface for image upload
- Training history visualization
- Model architecture display
"""

# Standard library imports
import os
import json
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

# Local application imports
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.predict import PredictionPipeline
from src.cnnClassifier.entity.config_entity import TrainingConfig
from src.cnnClassifier.components.training import Training

# Initialize Flask application
app = Flask(__name__)
CORS(app)

class ClientApp:
    """
    Client application class that handles image processing and prediction.
    
    Attributes:
        filename (str): Name of the input image file
        classifier (PredictionPipeline): Instance of the prediction pipeline
    """
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# Web Interface Routes
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """Render the main page of the application."""
    return render_template('index.html')

# API Endpoints
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    Handle image prediction requests.
    
    Expects a JSON payload with base64 encoded image data.
    Returns the prediction result or error message.
    """
    if 'image' not in request.json:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.json['image']
    
    try:
        decodeImage(image, clApp.filename)
        result = clApp.classifier.predict()
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_training_history():
    """
    Display model training history and performance metrics.
    
    Generates or loads training history data and creates visualizations
    of model performance metrics including accuracy and loss curves.
    """
    history_path = os.path.join('static', 'history.json')
    
    if not os.path.exists(history_path):
        # Initialize training configuration
        config = TrainingConfig(
            root_dir=Path('artifacts/training'),
            trained_model_path=Path('artifacts/training/model.keras'),
            updated_base_model_path=Path('artifacts/training/updated_base_model'),
            training_data=Path('artifacts/data_ingestion/poultry_diseases'),
            params_epochs=10,
            params_batch_size=32,
            params_is_augmentation=True,
            params_image_size=[224, 224, 3]
        )

        # Train model and generate performance metrics
        training_instance = Training(config)
        training_instance.train_valid_generator()
        history_data = training_instance.train(callback_list=[])

        # Save training history
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)

        # Generate accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['accuracy'], label='Training Accuracy')
        plt.plot(history_data['epochs'], history_data['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Model Accuracy')
        plt.savefig(os.path.join('static', 'training_accuracy.png'))

        # Generate loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['loss'], label='Training Loss')
        plt.plot(history_data['epochs'], history_data['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.savefig(os.path.join('static', 'training_loss.png'))
        plt.close()

        # Generate model architecture visualization
        plot_model(training_instance.model, 
                  to_file=os.path.join('static', 'model_architecture.png'),
                  show_shapes=True,
                  show_layer_names=True)
    else:
        # Load existing history data
        with open(history_path, 'r') as f:
            history_data = json.load(f)

    return render_template('history.html', history_data=history_data)

if __name__ == "__main__":
    # Initialize client application and start server
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
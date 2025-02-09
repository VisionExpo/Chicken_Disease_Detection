from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage  # Ensure this is the correct import
from cnnClassifier.pipeline.predict import PredictionPipeline
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.components.training import Training  # Ensure this is the correct import
from pathlib import Path
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import json

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

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

@app.route('/history', methods=['GET'])
def get_training_history():
    history_path = os.path.join('static', 'history.json')
    
    if not os.path.exists(history_path):
        # Initialize the config with the correct paths and parameters
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

        # Initialize the Training class with the config object
        training_instance = Training(config)

        # Set up the data generators
        training_instance.train_valid_generator()

        # Start the training process and get the history
        history_data = training_instance.train(callback_list=[])

        # Save the training history as a JSON file
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=4)

        # Plot training accuracy and loss as static images
        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['accuracy'], label='Training Accuracy')
        plt.plot(history_data['epochs'], history_data['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Model Accuracy')
        plt.savefig(os.path.join('static', 'training_accuracy.png'))

        plt.figure(figsize=(10, 5))
        plt.plot(history_data['epochs'], history_data['loss'], label='Training Loss')
        plt.plot(history_data['epochs'], history_data['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Model Loss')
        plt.savefig(os.path.join('static', 'training_loss.png'))
        plt.close()

        # Save model architecture as a static image
        plot_model(training_instance.model, to_file=os.path.join('static', 'model_architecture.png'), show_shapes=True, show_layer_names=True)

    else:
        # Load existing history data from the file
        with open(history_path, 'r') as f:
            history_data = json.load(f)

    # Return the history data as JSON and render the template with the history data
    return render_template('history.html', history_data=history_data)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

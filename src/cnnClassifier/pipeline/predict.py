import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.keras"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)

        # Example accuracy data (replace with actual model accuracy)
        accuracy = [0.6, 0.7, 0.8, 0.85, 0.9]  # Training accuracy
        validation_accuracy = [0.55, 0.65, 0.75, 0.8, 0.85]  # Validation accuracy

        if result[0] == 0:
            prediction = 'Coccidiosis'
        elif result[0] == 1:
            prediction = 'Healthy'
        elif result[0] == 2:
            prediction = 'NCD'
        elif result[0] == 3:
            prediction = 'Salmonella'
        else:
            prediction = 'Unknown'

        return {
            "prediction": prediction,
            "accuracy": accuracy,
            "validation_accuracy": validation_accuracy
        }

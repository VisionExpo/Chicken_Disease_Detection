import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        try:
            # Load model
            model_path = os.path.join("artifacts", "training", "model.keras")
            if not os.path.exists(model_path):
                return {"error": "Model file does not exist."}
            model = load_model(model_path)

        except Exception as e:
            return {"error": f"Model loading failed: {str(e)}"}

        try:
            imagename = self.filename
            if not os.path.exists(imagename):
                return {"error": "Image file does not exist."}
            test_image = image.load_img(imagename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            print("Input image shape:", test_image.shape)  # Debugging statement to log input image shape
            predictions = model.predict(test_image)
            result = np.argmax(predictions, axis=1)
            print("Model predictions:", predictions)  # Debugging statement to log model predictions

        except Exception as e:
            return {"error": f"Image processing failed: {str(e)}"}

        # Define accuracy and validation_accuracy
        expected_labels = [0, 1, 2, 3]  # Define actual expected labels based on your dataset
        expected_validation_labels = [0, 1, 2, 3]  # Define actual expected validation labels

        print("Predictions:", predictions)  # Debugging statement to log predictions
        print("Expected Labels:", expected_labels)  # Debugging statement to log expected labels
        accuracy = np.mean(predictions == expected_labels)  # Calculate accuracy
        validation_accuracy = np.mean(predictions == expected_validation_labels)  # Calculate validation accuracy

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

        response = {
            "prediction": prediction,
            "accuracy": accuracy,  # Return calculated accuracy
            "validation_accuracy": validation_accuracy
        }
        
        print("Response being sent:", response)  # Debugging statement to log the response
        return response

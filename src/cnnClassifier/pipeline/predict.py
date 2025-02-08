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
            model_path = os.path.join("artifacts", "training", "model.keras")  # Path to the trained model

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

            predictions = model.predict(test_image)
            result = np.argmax(predictions, axis=1)
            confidence = float(np.max(predictions))  # Get the confidence score

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
                "confidence": round(confidence * 100, 2)  # Convert to percentage and round to 2 decimal places
            }
            return response

        except Exception as e:
            return {"error": f"Image processing failed: {str(e)}"}
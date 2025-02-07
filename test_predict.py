import requests
import base64

# URL of the local server
url = "http://127.0.0.1:8080/predict"

# Load the image file
with open("inputImage.jpg", "rb") as img_file:
    # Encode the image to base64
    image_data = base64.b64encode(img_file.read()).decode('utf-8')

# Prepare the payload
payload = {
    "image": image_data  # Use the base64 string
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print("Response:", response.json())

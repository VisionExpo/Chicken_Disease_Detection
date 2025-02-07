import requests
import base64

# Path to the image file
image_path = 'inputImage.jpg'  # Update this path as necessary

# Read the image and convert it to base64
with open(image_path, 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Prepare the payload
payload = {
    "image": encoded_string
}

# Send a POST request to the /predict route
response = requests.post('http://127.0.0.1:8080/predict', json=payload)

# Print the response
print(response.json())

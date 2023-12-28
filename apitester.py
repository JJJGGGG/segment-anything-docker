from base64 import b64decode, b64encode
import json
import requests
import gdown

gdown.download("https://drive.google.com/uc?id=1VIiMg7_AEBIW8gJmG5kOLz8eOdEGLfhP")

with open("can.jpg", 'rb') as file:
    image = file.read()

body = {
    "image": b64encode(image).decode(),
    "box": [883, 749, 2000, 3100]
}

URL = "http://[::]:80"

r = requests.post(f"{URL}/segment", json=body, headers={"Content-Type": "application/json"})


res = r.json()

print(res)

f = b64decode(res["encoded_binary_mask"])

with open("response.png", 'wb') as file:
    file.write(f)
import torch
from torchvision import models, transforms
from PIL import Image
import json
import requests
import easyocr
import cv2
import numpy as np
import mysql.connector
# from ocr import run_ocr  # Not used in this integrated script

# Load ResNet model for object recognition
model = models.resnet50(pretrained=True)
model.eval()

# Load ImageNet labels
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
response = requests.get(LABELS_URL)
labels_map = {int(k): v[1] for k, v in json.loads(response.text).items()}

# OCR Initialization
reader = easyocr.Reader(['en'])

# MySQL Database Connection
conn = mysql.connector.connect(
    host="localhost",
    user="karthik",      # Change this if needed
    password="123456",   # Change this if needed
    database="ocr_db"
)
cursor = conn.cursor()

# Ensure table exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS extracted_text (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_name VARCHAR(255),
        extracted_text TEXT
    )
""")
conn.commit()

# Image Preprocessing for Object Recognition
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# Perform Object Recognition
def detect_object(image_path):
    img_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
    predicted_class = labels_map[outputs.argmax().item()]
    print("Detected Object:", predicted_class)
    # Modified detection condition: Accept additional related labels
    if ("wheel" in predicted_class.lower() or
        predicted_class.lower() in ["freight_car", "train", "french_horn"]):
        return True, predicted_class
    return False, predicted_class

# Perform OCR
def perform_ocr(image_path):
    img = cv2.imread(image_path)
    result = reader.readtext(img)
    extracted_text = "\n".join([detection[1] for detection in result])
    cursor.execute("INSERT INTO extracted_text (image_name, extracted_text) VALUES (%s, %s)",
                   (image_path, extracted_text))
    conn.commit()
    print(f"Extracted text saved to database: {extracted_text}")

# Main function
def main(image_path):
    detected, predicted_class = detect_object(image_path)
    if detected:
        print("Railway wheel detected! Running OCR...")
        perform_ocr(image_path)
    else:
        print("No railway wheel detected. Skipping OCR.")

if __name__ == "__main__":
    image_path = r"C:\Users\rsk72\Desktop\23e12fce-1e0f-47e9-b26e-ff3f1277ab22.jpg"  # Change to your actual image path
    main(image_path)
    # Close DB Connection after main execution
    cursor.close()
    conn.close()

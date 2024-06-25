import os
import torch
from PIL import Image
from model import pretrain_model, test_transform, Labels

UPLOAD_FOLDER = 'static/uploads/'

def predict(image_path):
    try:
        image = Image.open(image_path)
        
        # Convert RGBA image to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        print("Image format:", image.format)
        print("Image mode:", image.mode)
        print("Image size:", image.size)

        image_tensor = test_transform(image).unsqueeze(0).to('cpu')
        print("Image tensor shape:", image_tensor.shape)

        pretrain_model.eval()
        with torch.no_grad():
            output = pretrain_model(image_tensor)
            _, pred = torch.max(output, 1)
            label = Labels[pred.item()]
            print("Predicted label:", label)
        return label
    except Exception as e:
        print("Error:", e)
        return "Error"

if __name__ == "__main__":
    image_path = "static/uploads/capture.png"  # Replace with the path to your image
    prediction = predict(image_path)

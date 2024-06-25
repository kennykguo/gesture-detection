import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# https://github.com/Minhhnh/Deploy-hand-gesture-recognitioan-model-with-Flask/tree/main

device = torch.device('cpu')

Labels = { 
    0: 'ClickMode',
    1: 'Continue',
    2: 'Fan',
    3: 'Light',
    4: 'Off',
    5: 'On',
    6: 'One',
    7: 'Stop',
    8: 'Two',
}

test_transform = transforms.Compose([
    transforms.Resize((240, 240)),  # Resize the image to match the expected input size
    transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    transforms.Normalize([0.5345, 0.5550, 0.5419], [0.2360, 0.2502, 0.2615])  # Normalize the image
])

pretrain_model = models.resnet18()
pretrain_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 9))
path_model_pretrain = "hand_model18.pt"
pretrain_model.load_state_dict(torch.load(path_model_pretrain, map_location=device), strict=False)
pretrain_model.to(device)
pretrain_model.eval()


def predict(image_path):
    try:
        image = Image.open(image_path)

        # Convert RGBA image to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image = test_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = pretrain_model(image)
            _, pred = torch.max(output, 1)
            label = Labels[pred.item()]
        return label
    except Exception as e:
        print("Error:", e)
        return "Error"
"""Predict the flower species in one image.

Usage:
    python src/predict.py path/to/flower.jpg
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "best_model.pth"
CLASSES = ROOT / "class_names.json"

# Same preprocessing as validation/test during training
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    """Build MobileNetV2 with a 102-class head and load the trained weights."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 102)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


def predict(image_path, top_k=3, model=None):
    """Return a list of (flower_name, probability) for the top_k classes."""
    model = model or load_model()
    class_names = json.loads(CLASSES.read_text())
    image = Image.open(image_path).convert("RGB")
    x = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    values, indices = probs.topk(top_k)
    return [(class_names[i], float(v)) for v, i in zip(values, indices)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/predict.py path/to/flower.jpg")
    for name, p in predict(sys.argv[1]):
        print(f"{name}: {p * 100:.1f}%")

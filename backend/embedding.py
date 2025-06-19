import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

def load_dinov2_model(device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    model.eval()
    return model, device

def get_transform():
    return T.Compose([
        T.Resize(244),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def extract_embedding(model, image: Image.Image, device):
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding[0].cpu().numpy() 
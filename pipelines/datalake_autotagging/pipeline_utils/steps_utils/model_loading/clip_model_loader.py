import logging

import torch
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


def get_device(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    elif device.startswith("cuda") and not torch.cuda.is_available():
        print("Using CPU")
        device = torch.device("cpu")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    return device


def clip_load_model(model_name: str, device: str):
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(get_device(device=device))
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

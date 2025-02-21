import logging

from ultralytics import YOLO
import torch

logger = logging.getLogger(__name__)


def ultralytics_load_model(weights_path_to_load: str, device: str) -> YOLO:
    """
    Loads a YOLO model from the given weights file and moves it to the specified device.

    This function loads a YOLO model using the provided weights path and transfers it
    to the specified device (e.g., 'cpu' or 'cuda'). It raises an error if the weights
    file is not found or cannot be loaded.

    Args:
        weights_path_to_load (str): The file path to the YOLO model weights.
        device (str): The device to which the model should be moved ('cpu' or 'cuda').

    Returns:
        YOLO: The loaded YOLO model ready for inference or training.

    Raises:
        RuntimeError: If the weights file cannot be loaded or the device is unavailable.
    """
    loaded_model = YOLO(weights_path_to_load)
    torch_device = torch.device(device)
    logger.info(f"Loading model on device: {torch_device}")
    loaded_model.to(device=device)
    return loaded_model

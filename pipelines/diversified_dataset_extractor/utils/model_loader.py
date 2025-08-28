from abc import abstractmethod
from enum import Enum, auto
from typing import Any

import numpy as np
from open_clip.model import CLIP
from PIL import Image
from torch._C._te import Tensor
from torchvision.transforms import transforms


class SupportedEmbeddingModels(Enum):
    OPENCLIP = auto()


class EmbeddingModel:
    def __init__(self, device: str):
        self.device = device

    @abstractmethod
    def apply_preprocessing(self, image: Image) -> Any:
        pass

    @abstractmethod
    def encode_image(self, image: Image) -> np.ndarray:
        pass


class OpenClipEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model: CLIP,
        preprocessing: transforms.Compose,
        device: str,
    ):
        super().__init__(device=device)
        self.model = model
        self.preprocessing = preprocessing
        self.device = device

    def apply_preprocessing(self, image: Image) -> Tensor:
        return self.preprocessing(image).unsqueeze(0).to(self.device)

    def encode_image(self, image: Image) -> np.ndarray:
        return self.model.encode_image(image).detach().cpu().numpy()


def is_embedding_model_name_valid(
    source: str, target: SupportedEmbeddingModels
) -> bool:
    """
    Check if the provided embedding model is valid.

    Args:
        source: The provided embedding model.
        target: The target embedding model.

    Returns:
        If the provided embedding model is valid.
    """
    return source.upper() == target.name

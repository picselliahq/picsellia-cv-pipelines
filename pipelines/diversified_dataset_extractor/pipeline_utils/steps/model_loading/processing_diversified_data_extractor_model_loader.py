from abc import abstractmethod
from enum import Enum, auto
from typing import Any

import numpy as np
import open_clip
import torch
from open_clip.model import CLIP
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from PIL import Image
from torch._C._te import Tensor
from torchvision.transforms import transforms

from diversified_dataset_extractor.pipeline_utils.parameters.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)


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


@step
def load_diversified_data_extractor_model(pretrained_weights: str) -> EmbeddingModel:
    context: PicselliaProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model_name = context.processing_parameters.embedding_model

    if is_embedding_model_name_valid(
        source=embedding_model_name, target=SupportedEmbeddingModels.OPENCLIP
    ):
        model_architecture = context.processing_parameters.model_architecture

        (
            model,
            _,
            preprocessing_transformations,
        ) = open_clip.create_model_and_transforms(
            model_name=model_architecture,
            pretrained=pretrained_weights,
        )

        model.to(device)
        embedding_model = OpenClipEmbeddingModel(
            model=model,
            preprocessing=preprocessing_transformations,
            device=device,
        )

    else:
        raise ValueError(
            f"The provided model '{context.processing_parameters.embedding_model}' is not supported yet. "
            f"Supported models are {[member.name.lower() for member in SupportedEmbeddingModels]}."
        )

    return embedding_model

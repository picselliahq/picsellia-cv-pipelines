import open_clip

from src.picsellia_cv_engine import Pipeline, step
from src.picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.parameters.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps.model_loading.processing_diversified_data_extractor_model_loader import (
    SupportedEmbeddingModels,
    is_embedding_model_name_valid,
)


def validate_pretrained_weights(
    pretrained_weights: str, available_pretrained_weights: str
) -> None:
    """
    Validate the provided pretrained weights.

    Args:
        pretrained_weights: The provided pretrained weights.
        available_pretrained_weights: The available pretrained weights.

    Raises:
        ValueError: If the provided pretrained weights are not available.
    """
    if pretrained_weights not in available_pretrained_weights:
        raise ValueError(
            f"The provided pretrained weights '{pretrained_weights}' are not available. "
            f"Available pretrained weights are {available_pretrained_weights}."
        )


def validate_model_architecture(
    model_architecture: str, available_model_names: str
) -> None:
    """
    Validate the provided model architecture.

    Args:
        model_architecture: The provided model architecture.
        available_model_names: The available model names.

    Raises:
        ValueError: If the provided model architecture is not available.
        NotImplementedError: If the provided model architecture is a HuggingFace model.
    """
    if model_architecture not in available_model_names:
        raise ValueError(
            f"The provided model '{model_architecture}' is not available. "
            f"Available models are {available_model_names}."
        )
    elif model_architecture.startswith(open_clip.factory.HF_HUB_PREFIX):
        raise NotImplementedError(
            f"The provided model '{model_architecture}' is a "
            f"HuggingFace model and is not supported yet. "
            f"Please provide a model from the list of available models: {available_model_names}."
        )


@step
def validate_diversified_data_extractor_weights() -> str:
    context: PicselliaProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    embedding_model_name = context.processing_parameters.embedding_model
    model_architecture = context.processing_parameters.model_architecture
    pretrained_weights = context.processing_parameters.pretrained_weights

    if is_embedding_model_name_valid(
        source=embedding_model_name, target=SupportedEmbeddingModels.OPENCLIP
    ):
        validate_model_architecture(
            model_architecture=model_architecture,
            available_model_names=open_clip.list_models(),
        )
        validate_pretrained_weights(
            pretrained_weights=pretrained_weights,
            available_pretrained_weights=open_clip.pretrained.list_pretrained_tags_by_model(
                model=model_architecture
            ),
        )

        return pretrained_weights

    else:
        raise ValueError(
            f"The provided model '{context.processing_parameters.embedding_model}' is not supported yet. "
            f"Supported models are {[member.name.lower() for member in SupportedEmbeddingModels]}."
        )

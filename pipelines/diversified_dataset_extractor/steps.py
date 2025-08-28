import open_clip
import torch
from picsellia_cv_engine.core.contexts.processing.dataset import (
    PicselliaDatasetProcessingContext,
)
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.data_validator import (
    ProcessingDiversifiedDataExtractorDataValidator,
)
from utils.model_loader import (
    EmbeddingModel,
    OpenClipEmbeddingModel,
    SupportedEmbeddingModels,
    is_embedding_model_name_valid,
)
from utils.model_validator import (
    validate_model_architecture,
    validate_pretrained_weights,
)
from utils.parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from utils.processing import (
    DiversifiedDataExtractorProcessing,
)


@step
def validate_data(
    dataset: TBaseDataset,
) -> None:
    validator = ProcessingDiversifiedDataExtractorDataValidator(
        dataset=dataset,
    )
    validator.validate()


@step
def load_model(pretrained_weights: str) -> EmbeddingModel:
    context: PicselliaDatasetProcessingContext[
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


@step
def process(
    input_dataset: TBaseDataset,
    output_dataset: TBaseDataset,
    embedding_model: EmbeddingModel,
):
    context: PicselliaDatasetProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    processor = DiversifiedDataExtractorProcessing(
        client=context.client,
        datalake=context.client.get_datalake(),
        input_dataset=input_dataset,
        output_dataset_version=output_dataset.dataset_version,
        embedding_model=embedding_model,
        distance_threshold=context.processing_parameters.distance_threshold,
    )
    processor.process()


@step
def validate_weights() -> str:
    context: PicselliaDatasetProcessingContext[
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

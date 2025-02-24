from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.steps.data_extraction.processing_data_extractor import (
    get_processing_dataset_context,
)

from pipelines.diversified_dataset_extractor.pipeline_utils.parameters.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps.data_validation.processing_diversified_data_extractor_data_validator import (
    validate_diversified_data_extractor_data,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps.model_loading.processing_diversified_data_extractor_model_loader import (
    load_diversified_data_extractor_model,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps.processing.diversified_data_extractor_processing import (
    process,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps.weights_validation.processing_diversified_data_extractor_weights_validator import (
    validate_diversified_data_extractor_weights,
)


def get_context() -> PicselliaProcessingContext[
    ProcessingDiversifiedDataExtractorParameters
]:
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingDiversifiedDataExtractorParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def diversified_data_extractor_pipeline() -> None:
    dataset_context = get_processing_dataset_context(skip_asset_listing=True)

    validate_diversified_data_extractor_data(dataset_context=dataset_context)
    pretrained_weights = validate_diversified_data_extractor_weights()
    embedding_model = load_diversified_data_extractor_model(
        pretrained_weights=pretrained_weights
    )

    process(dataset_context=dataset_context, embedding_model=embedding_model)


if __name__ == "__main__":
    diversified_data_extractor_pipeline()

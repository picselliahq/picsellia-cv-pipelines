from picsellia_cv_engine.core.contexts import PicselliaProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets

from diversified_dataset_extractor.pipeline_utils.parameters.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from diversified_dataset_extractor.pipeline_utils.steps.data_validation.processing_diversified_data_extractor_data_validator import (
    validate_diversified_data_extractor_data,
)
from diversified_dataset_extractor.pipeline_utils.steps.model_loading.processing_diversified_data_extractor_model_loader import (
    load_diversified_data_extractor_model,
)
from diversified_dataset_extractor.pipeline_utils.steps.processing.diversified_data_extractor_processing import (
    process,
)
from diversified_dataset_extractor.pipeline_utils.steps.weights_validation.processing_diversified_data_extractor_weights_validator import (
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
    dataset = load_coco_datasets(skip_asset_listing=True)

    validate_diversified_data_extractor_data(dataset=dataset)
    pretrained_weights = validate_diversified_data_extractor_weights()
    embedding_model = load_diversified_data_extractor_model(
        pretrained_weights=pretrained_weights
    )

    process(dataset=dataset, embedding_model=embedding_model)


if __name__ == "__main__":
    diversified_data_extractor_pipeline()

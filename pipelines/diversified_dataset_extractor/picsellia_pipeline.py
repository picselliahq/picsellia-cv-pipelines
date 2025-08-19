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
from picsellia_cv_engine.core.services.utils.picsellia_context import (
    create_picsellia_dataset_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets

context = create_picsellia_dataset_processing_context(
    processing_parameters_cls=ProcessingDiversifiedDataExtractorParameters,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def diversified_data_extractor_pipeline() -> None:
    datasets = load_coco_datasets(skip_asset_listing=True)

    validate_diversified_data_extractor_data(dataset=datasets["input"])
    pretrained_weights = validate_diversified_data_extractor_weights()
    embedding_model = load_diversified_data_extractor_model(
        pretrained_weights=pretrained_weights
    )

    process(
        input_dataset=datasets["input"],
        output_dataset=datasets["output"],
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    diversified_data_extractor_pipeline()

# type: ignore

from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.steps.data_extraction.processing_data_extractor import (
    get_processing_dataset_collection,
)
from picsellia_cv_engine.steps.data_upload.classification_dataset_context_uploader import (
    upload_classification_dataset_context,
)

from pipelines.bounding_box_cropper.pipeline_utils.parameters.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from pipelines.bounding_box_cropper.pipeline_utils.steps.data_validation.processing_bounding_box_cropper_data_validator import (
    validate_bounding_box_cropper_data,
)
from pipelines.bounding_box_cropper.pipeline_utils.steps.processing.bounding_box_cropper_processing import (
    process,
)


def get_context() -> PicselliaProcessingContext[ProcessingBoundingBoxCropperParameters]:
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingBoundingBoxCropperParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def bounding_box_cropper_processing_pipeline() -> None:
    dataset_collection = get_processing_dataset_collection()
    validate_bounding_box_cropper_data(dataset_context=dataset_collection["input"])
    output_dataset_context = process(dataset_collection=dataset_collection)
    upload_classification_dataset_context(dataset_context=output_dataset_context)


if __name__ == "__main__":
    bounding_box_cropper_processing_pipeline()

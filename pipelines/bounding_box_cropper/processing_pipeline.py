# type: ignore

from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset

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
    dataset_collection = load_coco_datasets()
    validate_bounding_box_cropper_data(dataset=dataset_collection["input"])
    output_dataset = process(dataset_collection=dataset_collection)
    upload_full_dataset(dataset=output_dataset)


if __name__ == "__main__":
    bounding_box_cropper_processing_pipeline()

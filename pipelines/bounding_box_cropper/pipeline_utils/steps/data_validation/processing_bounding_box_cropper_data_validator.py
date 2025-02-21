from src.picsellia_cv_engine import Pipeline
from src.picsellia_cv_engine import step
from src.picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from pipelines.bounding_box_cropper.pipeline_utils.parameters.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from pipelines.bounding_box_cropper.pipeline_utils.steps_utils.data_validation.processing_bounding_box_cropper_data_validator import (
    ProcessingBoundingBoxCropperDataValidator,
)


@step
def validate_bounding_box_cropper_data(
    dataset_context: CocoDatasetContext,
) -> CocoDatasetContext:
    """
    Validates the dataset for the bounding box cropping process.

    This function retrieves the active processing context and validates the provided dataset context
    based on the parameters of the bounding box cropping task. It uses the `ProcessingBoundingBoxCropperDataValidator`
    to perform the validation, ensuring that the dataset is suitable for processing (e.g., checking for
    correct labels, annotations, etc.). The validated dataset context is then returned.

    Args:
        dataset_context (DatasetContext): The dataset context to be validated.

    Returns:
        DatasetContext: The validated dataset context, ready for further processing.
    """
    context: PicselliaProcessingContext[ProcessingBoundingBoxCropperParameters] = (
        Pipeline.get_active_context()
    )

    validator = ProcessingBoundingBoxCropperDataValidator(
        dataset_context=dataset_context,
        client=context.client,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
        datalake=context.processing_parameters.datalake,
        fix_annotation=context.processing_parameters.fix_annotation,
    )
    dataset_context = validator.validate()
    return dataset_context

from picsellia_cv_engine.core import (
    CocoDataset,
)
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from bounding_box_cropper.pipeline_utils.parameters.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from bounding_box_cropper.pipeline_utils.steps_utils.data_validation.processing_bounding_box_cropper_data_validator import (
    ProcessingBoundingBoxCropperDataValidator,
)


@step
def validate_bounding_box_cropper_data(
    dataset: CocoDataset,
) -> CocoDataset:
    """
    Validates the dataset for the bounding box cropping process.

    This function retrieves the active processing context and validates the provided dataset
    based on the parameters of the bounding box cropping task. It uses the `ProcessingBoundingBoxCropperDataValidator`
    to perform the validation, ensuring that the dataset is suitable for processing (e.g., checking for
    correct labels, annotations, etc.). The validated dataset is then returned.

    Args:
        dataset (Dataset): The dataset to be validated.

    Returns:
        Dataset: The validated dataset, ready for further processing.
    """
    context: PicselliaProcessingContext[ProcessingBoundingBoxCropperParameters] = (
        Pipeline.get_active_context()
    )

    validator = ProcessingBoundingBoxCropperDataValidator(
        dataset=dataset,
        client=context.client,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
        datalake=context.processing_parameters.datalake,
        fix_annotation=context.processing_parameters.fix_annotation,
    )
    dataset = validator.validate()
    return dataset

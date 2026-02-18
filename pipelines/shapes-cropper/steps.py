from picsellia_cv_engine.core import (
    CocoDataset,
    DatasetCollection,
)
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_context import (
    PicselliaDatasetProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.data_validator import ProcessingShapesCropperDataValidator
from utils.parameters import ProcessingShapesCropperParameters
from utils.processing import ShapesCropperProcessing


@step
def process(
    dataset_collection: DatasetCollection[CocoDataset],
) -> CocoDataset:
    context: PicselliaDatasetProcessingContext[
        ProcessingShapesCropperParameters
    ] = Pipeline.get_active_context()

    processor = ShapesCropperProcessing(
        dataset_collection=dataset_collection,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
    )
    dataset_collection = processor.process()
    return dataset_collection["output"]


@step
def validate_shapes_cropper_data(
    dataset: CocoDataset,
) -> CocoDataset:
    """
    Validates the dataset for the shapes cropping process.

    This function retrieves the active processing context and validates the provided dataset
    based on the parameters of the shapes cropping task. It uses the `ProcessingShapesCropperDataValidator`
    to perform the validation, ensuring that the dataset is suitable for processing (e.g., checking for
    correct labels, annotations, etc.). The validated dataset is then returned.

    Args:
        dataset (Dataset): The dataset to be validated.

    Returns:
        Dataset: The validated dataset, ready for further processing.
    """
    context: PicselliaDatasetProcessingContext[
        ProcessingShapesCropperParameters
    ] = Pipeline.get_active_context()

    validator = ProcessingShapesCropperDataValidator(
        dataset=dataset,
        client=context.client,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
        datalake=context.processing_parameters.datalake,
        fix_annotation=context.processing_parameters.fix_annotation,
    )
    dataset = validator.validate()
    return dataset

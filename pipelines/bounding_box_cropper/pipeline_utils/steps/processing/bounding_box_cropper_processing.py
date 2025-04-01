from picsellia_cv_engine.core import (
    CocoDataset,
    DatasetCollection,
)
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.bounding_box_cropper.pipeline_utils.parameters.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from pipelines.bounding_box_cropper.pipeline_utils.steps_utils.processing.bounding_box_cropper_processing import (
    BoundingBoxCropperProcessing,
)


@step
def process(
    dataset_collection: DatasetCollection[CocoDataset],
) -> CocoDataset:
    context: PicselliaProcessingContext[ProcessingBoundingBoxCropperParameters] = (
        Pipeline.get_active_context()
    )

    processor = BoundingBoxCropperProcessing(
        dataset_collection=dataset_collection,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
    )
    dataset_collection = processor.process()
    return dataset_collection["output"]

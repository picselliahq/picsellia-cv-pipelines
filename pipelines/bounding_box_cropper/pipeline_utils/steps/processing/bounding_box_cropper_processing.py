from src.picsellia_cv_engine import Pipeline
from src.picsellia_cv_engine import step
from src.picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from src.picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from pipelines.bounding_box_cropper.pipeline_utils.parameters.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from pipelines.bounding_box_cropper.pipeline_utils.steps_utils.processing.bounding_box_cropper_processing import (
    BoundingBoxCropperProcessing,
)


@step
def process(
    dataset_collection: DatasetCollection[CocoDatasetContext],
) -> CocoDatasetContext:
    context: PicselliaProcessingContext[ProcessingBoundingBoxCropperParameters] = (
        Pipeline.get_active_context()
    )

    processor = BoundingBoxCropperProcessing(
        dataset_collection=dataset_collection,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
    )
    dataset_collection = processor.process()
    return dataset_collection["output"]

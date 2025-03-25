from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.data.dataset.coco_dataset import (
    CocoDataset,
)
from picsellia_cv_engine.models.data.dataset.dataset_collection import (
    DatasetCollection,
)

from pipelines.dataset_tiler.pipeline_utils.parameters.processing_tiler_parameters import (
    ProcessingTilerParameters,
)
from pipelines.dataset_tiler.pipeline_utils.steps_utils.processing.tiler_processing_factory import (
    TilerProcessingFactory,
)


@step
def process(
    dataset_collection: DatasetCollection[CocoDataset],
) -> CocoDataset:
    context: PicselliaProcessingContext[ProcessingTilerParameters] = (
        Pipeline.get_active_context()
    )

    processor = TilerProcessingFactory.create_tiler_processing(
        dataset_type=dataset_collection["input"].dataset_version.type,
        tile_height=context.processing_parameters.tile_height,
        tile_width=context.processing_parameters.tile_width,
        overlap_height_ratio=context.processing_parameters.overlap_height_ratio,
        overlap_width_ratio=context.processing_parameters.overlap_width_ratio,
        min_annotation_area_ratio=context.processing_parameters.min_annotation_area_ratio,
        min_annotation_width=context.processing_parameters.min_annotation_width,
        min_annotation_height=context.processing_parameters.min_annotation_height,
        tiling_mode=context.processing_parameters.tiling_mode,
        padding_color_value=context.processing_parameters.padding_color_value,
    )

    dataset_collection = processor.process_dataset_collection(
        dataset_collection=dataset_collection
    )

    return dataset_collection["output"]

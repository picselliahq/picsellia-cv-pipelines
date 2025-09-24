from picsellia.types.enums import InferenceType
from picsellia_cv_engine.core import (
    CocoDataset,
    DatasetCollection,
)
from picsellia_cv_engine.core.contexts.processing.dataset import (
    PicselliaDatasetProcessingContext,
)
from picsellia_cv_engine.core.services.data.dataset.validator import (
    NotConfiguredDatasetValidator,
)
from picsellia_cv_engine.core.services.data.dataset.validator.classification.coco_classification_dataset_context_validator import (
    CocoClassificationDatasetValidator,
)
from picsellia_cv_engine.core.services.data.dataset.validator.object_detection.coco_object_detection_dataset_validator import (
    CocoObjectDetectionDatasetValidator,
)
from picsellia_cv_engine.core.services.data.dataset.validator.segmentation.coco_segmentation_dataset_validator import (
    CocoSegmentationDatasetValidator,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.parameters import (
    ProcessingTilerParameters,
)
from utils.processing_tiler_data_validator import (
    ProcessingTilerDataValidator,
)
from utils.tiler_processing_factory import (
    TilerProcessingFactory,
)


@step
def validate_tiler_data(
    dataset: CocoDataset,
) -> CocoDataset:
    context: PicselliaDatasetProcessingContext[ProcessingTilerParameters] = (
        Pipeline.get_active_context()
    )

    # 1. First, perform dataset validation based on the dataset type.
    match dataset.dataset_version.type:
        case InferenceType.NOT_CONFIGURED:
            not_configured_dataset_validator = NotConfiguredDatasetValidator(
                dataset=dataset
            )
            not_configured_dataset_validator.validate()

        case InferenceType.SEGMENTATION:
            # Both object detection and segmentation dataset validators are used for segmentation datasets because,
            # within a COCO segmentation dataset, both the properties of bounding boxes and polygons are used.
            object_detection_dataset_validator = CocoObjectDetectionDatasetValidator(
                dataset=dataset,
                fix_annotation=context.processing_parameters.fix_annotation,
            )
            dataset = object_detection_dataset_validator.validate()

            segmentation_dataset_validator = CocoSegmentationDatasetValidator(
                dataset=dataset,
                fix_annotation=context.processing_parameters.fix_annotation,
            )
            dataset = segmentation_dataset_validator.validate()

        case InferenceType.OBJECT_DETECTION:
            object_detection_dataset_validator = CocoObjectDetectionDatasetValidator(
                dataset=dataset,
                fix_annotation=context.processing_parameters.fix_annotation,
            )
            dataset = object_detection_dataset_validator.validate()

        case InferenceType.CLASSIFICATION:
            classification_dataset_validator = CocoClassificationDatasetValidator(
                dataset=dataset,
            )
            dataset = classification_dataset_validator.validate()

        case _:
            raise ValueError(
                f"Dataset type {dataset.dataset_version.type} is not supported."
            )

    # 2. Then, perform validations specific to the tiler processing.
    processing_validator = ProcessingTilerDataValidator(
        client=context.client,
        tile_height=context.processing_parameters.tile_height,
        tile_width=context.processing_parameters.tile_width,
        overlap_height_ratio=context.processing_parameters.overlap_height_ratio,
        overlap_width_ratio=context.processing_parameters.overlap_width_ratio,
        min_annotation_area_ratio=context.processing_parameters.min_annotation_area_ratio,
        min_annotation_width=context.processing_parameters.min_annotation_width,
        min_annotation_height=context.processing_parameters.min_annotation_height,
        padding_color_value=context.processing_parameters.padding_color_value,
        datalake=context.processing_parameters.datalake,
    )
    processing_validator.validate()

    return dataset


@step
def process(
    dataset_collection: DatasetCollection[CocoDataset],
) -> CocoDataset:
    context: PicselliaDatasetProcessingContext[ProcessingTilerParameters] = (
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

from picsellia.types.enums import InferenceType
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.data.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.classification.coco_classification_dataset_context_validator import (
    CocoClassificationDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.common.not_configured_dataset_context_validator import (
    NotConfiguredDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.object_detection.coco_object_detection_dataset_context_validator import (
    CocoObjectDetectionDatasetContextValidator,
)
from picsellia_cv_engine.models.steps.data.dataset.validator.segmentation.coco_segmentation_dataset_context_validator import (
    CocoSegmentationDatasetContextValidator,
)

from pipelines.dataset_tiler.pipeline_utils.parameters.processing_tiler_parameters import (
    ProcessingTilerParameters,
)
from pipelines.dataset_tiler.pipeline_utils.steps_utils.data_validation.processing_tiler_data_validator import (
    ProcessingTilerDataValidator,
)


@step
def validate_tiler_data(
    dataset_context: CocoDatasetContext,
) -> CocoDatasetContext:
    context: PicselliaProcessingContext[ProcessingTilerParameters] = (
        Pipeline.get_active_context()
    )

    # 1. First, perform dataset validation based on the dataset type.
    match dataset_context.dataset_version.type:
        case InferenceType.NOT_CONFIGURED:
            not_configured_dataset_validator = NotConfiguredDatasetContextValidator(
                dataset_context=dataset_context
            )
            not_configured_dataset_validator.validate()

        case InferenceType.SEGMENTATION:
            # Both object detection and segmentation dataset validators are used for segmentation datasets because,
            # within a COCO segmentation dataset, both the properties of bounding boxes and polygons are used.
            object_detection_dataset_validator = (
                CocoObjectDetectionDatasetContextValidator(
                    dataset_context=dataset_context,
                    fix_annotation=context.processing_parameters.fix_annotation,
                )
            )
            dataset_context = object_detection_dataset_validator.validate()

            segmentation_dataset_validator = CocoSegmentationDatasetContextValidator(
                dataset_context=dataset_context,
                fix_annotation=context.processing_parameters.fix_annotation,
            )
            dataset_context = segmentation_dataset_validator.validate()

        case InferenceType.OBJECT_DETECTION:
            object_detection_dataset_validator = (
                CocoObjectDetectionDatasetContextValidator(
                    dataset_context=dataset_context,
                    fix_annotation=context.processing_parameters.fix_annotation,
                )
            )
            dataset_context = object_detection_dataset_validator.validate()

        case InferenceType.CLASSIFICATION:
            classification_dataset_validator = (
                CocoClassificationDatasetContextValidator(
                    dataset_context=dataset_context,
                )
            )
            dataset_context = classification_dataset_validator.validate()

        case _:
            raise ValueError(
                f"Dataset type {dataset_context.dataset_version.type} is not supported."
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

    return dataset_context

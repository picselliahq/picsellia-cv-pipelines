import json

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)

from pipelines.yolov8_classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)
from pipelines.yolov8_preannotation.pipeline_utils.steps_utils.processing.yolov8_preannotation_processing import (
    PreAnnotator,
    _check_model_type_sanity,
    _get_model_labels_name,
    _type_coherence_check,
)


@step
def process(
    model_context: UltralyticsModelContext, dataset_context: CocoDatasetContext
) -> CocoDatasetContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    _check_model_type_sanity(model_version=model_context.model_version)
    dataset_context.dataset_version = _type_coherence_check(
        dataset_version=dataset_context.dataset_version,
        model_version=model_context.model_version,
    )
    model_labels, model_infos = _get_model_labels_name(
        model_version=model_context.model_version
    )

    pre_annotator = PreAnnotator(
        client=context.client,
        dataset_version=dataset_context.dataset_version,
        model_context=model_context,
        model_labels=model_labels,
        parameters=context.processing_parameters,
    )

    pre_annotator.setup_preannotation_job()
    dataset_context.coco_data = pre_annotator.preannotate(
        confidence_threshold=context.processing_parameters.confidence_threshold
    )

    with open(dataset_context.coco_file_path, "w") as f:
        json.dump(dataset_context.coco_data, f)

    return dataset_context

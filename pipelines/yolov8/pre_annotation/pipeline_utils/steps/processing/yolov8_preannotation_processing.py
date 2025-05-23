import json
import os

from picsellia_cv_engine.core import (
    CocoDataset,
)
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel

from pipelines.yolov8.pre_annotation.pipeline_utils.steps_utils.processing.yolov8_preannotation_processing import (
    PreAnnotator,
    _check_model_type_sanity,
    _get_model_labels_name,
    _type_coherence_check,
)


@step
def process(model: UltralyticsModel, dataset: CocoDataset) -> CocoDataset:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    _check_model_type_sanity(model_version=model.model_version)
    dataset.dataset_version = _type_coherence_check(
        dataset_version=dataset.dataset_version,
        model_version=model.model_version,
    )
    model_labels, model_infos = _get_model_labels_name(
        model_version=model.model_version
    )

    pre_annotator = PreAnnotator(
        client=context.client,
        dataset_version=dataset.dataset_version,
        model=model,
        model_labels=model_labels,
        parameters=context.processing_parameters,
    )

    pre_annotator.setup_preannotation_job()
    dataset.coco_data = pre_annotator.preannotate(
        confidence_threshold=context.processing_parameters.confidence_threshold,
        agnostic_nms=context.processing_parameters.agnostic_nms,
    )

    if not dataset.coco_file_path:
        dataset.coco_file_path = os.path.join(
            dataset.annotations_dir, "coco_annotations.json"
        )
    with open(dataset.coco_file_path, "w") as f:
        json.dump(dataset.coco_data, f)

    return dataset

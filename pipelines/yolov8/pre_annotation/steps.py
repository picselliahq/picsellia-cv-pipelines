import json
import os
import os.path

from picsellia_cv_engine.core import (
    CocoDataset,
)
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_context import (
    PicselliaDatasetProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel
from utils.processing import (
    PreAnnotator,
    _check_model_type_sanity,
    _get_model_labels_name,
    _type_coherence_check,
)


@step
def get_model() -> UltralyticsModel:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()

    model_version = context.model_version
    model = UltralyticsModel(
        name=model_version.name,
        model_version=model_version,
        trained_weights_name=context.processing_parameters.model_file_name,
    )
    model.download_weights(destination_dir=os.path.join(context.working_dir, "model"))
    return model


@step
def load_model(model: UltralyticsModel, weights_path_to_load: str) -> UltralyticsModel:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        if weights_path_to_load.endswith(".onnx"):
            raise (
                ValueError(
                    "Cannot use ONNX model for preannotation, please use a .pt model"
                )
            )
        loaded_model = model.load_yolo_weights(
            weights_path=weights_path_to_load,
            device=context.processing_parameters.device,
        )
        model.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model


@step
def process(model: UltralyticsModel, dataset: CocoDataset) -> CocoDataset:
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()

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

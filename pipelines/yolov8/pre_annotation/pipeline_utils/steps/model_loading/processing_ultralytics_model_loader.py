import os.path

from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel
from picsellia_cv_engine.steps.ultralytics.model.loader import (
    load_yolo_weights,
)


@step
def load_processing_ultralytics_model(
    model: UltralyticsModel, weights_path_to_load: str
) -> UltralyticsModel:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        if weights_path_to_load.endswith(".onnx"):
            raise (
                ValueError(
                    "Cannot use ONNX model for preannotation, please use a .pt model"
                )
            )
        loaded_model = load_yolo_weights(
            weights_path_to_load=weights_path_to_load,
            device=context.processing_parameters.device,
        )
        model.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model

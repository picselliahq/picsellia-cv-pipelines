import os.path

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)

from pipelines.yolov8.training.classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps_utils.model_loading.ultralytics_model_context_loader import (
    ultralytics_load_model,
)


@step
def load_processing_ultralytics_model_context(
    model_context: UltralyticsModelContext, weights_path_to_load: str
) -> UltralyticsModelContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        if weights_path_to_load.endswith(".onnx"):
            raise (
                ValueError(
                    "Cannot use ONNX model for preannotation, please use a .pt model"
                )
            )
        loaded_model = ultralytics_load_model(
            weights_path_to_load=weights_path_to_load,
            device=context.processing_parameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model_context

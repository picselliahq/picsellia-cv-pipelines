import os.path

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)

from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model_context import (
    Yolov7ModelContext,
)
from pipelines.yolov8_classification.pipeline_utils.steps.model_loading.ultralytics_model_context_loader import (
    ultralytics_load_model,
)


@step
def yolov7_model_context_loader(
    model_context: Yolov7ModelContext, weights_path_to_load: str
) -> Yolov7ModelContext:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        loaded_model = ultralytics_load_model(
            weights_path_to_load=weights_path_to_load,
            device=context.hyperparameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model_context

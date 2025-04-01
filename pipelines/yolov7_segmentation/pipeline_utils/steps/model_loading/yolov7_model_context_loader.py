import os.path

from picsellia_cv_engine.core.contexts import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model_context import (
    Yolov7Model,
)
from pipelines.yolov8.training.pipeline_utils.steps_utils.model_loading.ultralytics_model_context_loader import (
    ultralytics_load_model,
)


@step
def yolov7_model_loader(model: Yolov7Model, weights_path_to_load: str) -> Yolov7Model:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        loaded_model = ultralytics_load_model(
            weights_path_to_load=weights_path_to_load,
            device=context.hyperparameters.device,
        )
        model.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model

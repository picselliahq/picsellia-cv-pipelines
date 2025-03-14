import os.path

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)

from pipelines.yolov8.training.classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)
from pipelines.yolov8.training.classification.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8.training.classification.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps_utils.model_loading.ultralytics_model_context_loader import (
    ultralytics_load_model,
)


@step
def load_ultralytics_model_context(
    model_context: UltralyticsModelContext, weights_path_to_load: str
) -> UltralyticsModelContext:
    """
    Loads an Ultralytics model from pretrained weights if available.

    This function retrieves the active training context and attempts to load the Ultralytics model from
    the pretrained weights specified in the model context. If the pretrained weights file exists, the model
    is loaded onto the specified device. If the pretrained weights are not found, a `FileNotFoundError` is raised.

    Args:
        model_context (ModelContext): The model context containing the path to the pretrained weights and
                                      other model-related configurations.
        weights_path_to_load (str): The path to the pretrained weights file to load the model from.

    Returns:
        ModelContext: The updated model context with the loaded model.

    Raises:
        FileNotFoundError: If the pretrained weights file is not found at the specified path in the model context.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

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

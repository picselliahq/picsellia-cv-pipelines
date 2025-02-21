import os
from typing import Optional

from src.picsellia_cv_engine import step, Pipeline
from src.picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model_context import (
    Yolov7ModelContext,
)


@step
def yolov7_model_context_extractor(
    pretrained_weights_name: Optional[str] = None,
    trained_weights_name: Optional[str] = None,
    config_name: Optional[str] = None,
    hyperparameters_name: Optional[str] = None,
    exported_weights_name: Optional[str] = None,
) -> Yolov7ModelContext:
    """
    Extracts a model context from the active Picsellia training experiment.

    This function retrieves the active training context from the pipeline and extracts the base model version
    from the experiment. It then creates a `ModelContext` object for the model, specifying the name and pretrained
    weights. The function downloads the necessary model weights to a specified directory and returns the
    initialized `ModelContext`.

    Returns:
        ModelContext: The extracted and initialized model context with the downloaded weights.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_version = context.experiment.get_base_model_version()
    model_context = Yolov7ModelContext(
        model_name=model_version.name,
        model_version=model_version,
        pretrained_weights_name=pretrained_weights_name,
        trained_weights_name=trained_weights_name,
        config_name=config_name,
        hyperparameters_name=hyperparameters_name,
        exported_weights_name=exported_weights_name,
    )
    model_context.download_weights(
        destination_path=os.path.join(os.getcwd(), context.experiment.name, "model")
    )
    model_context.set_hyperparameters_path(
        destination_path=os.path.join(
            os.getcwd(), context.experiment.name, "model", "weights"
        )
    )
    return model_context

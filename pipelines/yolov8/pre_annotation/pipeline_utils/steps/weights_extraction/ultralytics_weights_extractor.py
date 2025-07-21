import os

from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel


@step
def get_processing_ultralytics_model() -> UltralyticsModel:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_version = context.model_version
    model = UltralyticsModel(
        name=model_version.name,
        model_version=model_version,
        trained_weights_name=context.processing_parameters.model_file_name,
    )
    model.download_weights(destination_dir=os.path.join(context.working_dir, "model"))
    return model

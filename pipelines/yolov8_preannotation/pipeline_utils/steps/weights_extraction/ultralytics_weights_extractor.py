import os

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)

from pipelines.yolov8_classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)


@step
def get_processing_ultralytics_model_context() -> UltralyticsModelContext:
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_version = context.model_version
    model_context = UltralyticsModelContext(
        model_name=model_version.name,
        model_version=model_version,
        trained_weights_name=context.processing_parameters.model_file_name,
    )
    model_context.download_weights(
        destination_dir=os.path.join(os.getcwd(), context.job_id, "model")
    )
    return model_context

import logging

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)

from pipelines.yolov8.training.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8.training.pipeline_utils.steps_utils.model_export.ultralytics_model_context_exporter import (
    UltralyticsModelContextExporter,
)

logger = logging.getLogger(__name__)


@step
def export_ultralytics_model_context(model_context: UltralyticsModelContext):
    """
    Exports and saves the Ultralytics model context to the experiment.

    This function retrieves the active training context from the pipeline, exports the Ultralytics model
    in the specified format, and saves the exported model weights to the experiment. If the `exported_weights_dir`
    is not found in the model context, the export process is skipped, and a log message is generated.

    Args:
        model_context (ModelContext): The model context for the Ultralytics model to be exported and saved.

    Raises:
        If no `exported_weights_dir` is found in the model context, the export process is skipped, and
        a log message is generated.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_exporter = UltralyticsModelContextExporter(
        model_context=model_context
    )

    if model_context.exported_weights_dir:
        model_context_exporter.export_model_context(
            exported_model_destination_path=model_context.exported_weights_dir,
            export_format=context.export_parameters.export_format,
            hyperparameters=context.hyperparameters,
        )
        model_context_exporter.save_model_to_experiment(
            experiment=context.experiment,
            exported_weights_path=model_context.exported_weights_dir,
            exported_weights_name="model-latest",
        )
    else:
        logger.info(
            "No exported weights directory found in model context. Skipping export."
        )

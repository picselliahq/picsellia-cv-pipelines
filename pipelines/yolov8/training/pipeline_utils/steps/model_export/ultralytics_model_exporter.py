import logging

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.parameters import ExportParameters

from pipelines.yolov8.training.pipeline_utils.model.ultralytics_model import (
    UltralyticsModel,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8.training.pipeline_utils.steps_utils.model_export.ultralytics_model_context_exporter import (
    UltralyticsModelExporter,
)

logger = logging.getLogger(__name__)


@step
def export_ultralytics_model(model: UltralyticsModel):
    """
    Exports and saves the Ultralytics model to the experiment.

    This function retrieves the active training context from the pipeline, exports the Ultralytics model
    in the specified format, and saves the exported model weights to the experiment. If the `exported_weights_dir`
    is not found in the model, the export process is skipped, and a log message is generated.

    Args:
        model (Model): The model for the Ultralytics model to be exported and saved.

    Raises:
        If no `exported_weights_dir` is found in the model, the export process is skipped, and
        a log message is generated.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_exporter = UltralyticsModelExporter(model=model)

    if model.exported_weights_dir:
        model_exporter.export_model(
            exported_model_destination_path=model.exported_weights_dir,
            export_format=context.export_parameters.export_format,
            hyperparameters=context.hyperparameters,
        )
        model_exporter.save_model_to_experiment(
            experiment=context.experiment,
            exported_weights_path=model.exported_weights_dir,
            exported_weights_name="model-latest",
        )
    else:
        logger.info("No exported weights directory found in model. Skipping export.")

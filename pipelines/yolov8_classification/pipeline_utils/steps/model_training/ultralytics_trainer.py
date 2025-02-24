import os.path

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)

from pipelines.yolov8_classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)
from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8_classification.pipeline_utils.steps_utils.model_training.ultralytics_model_context_trainer import (
    UltralyticsModelContextTrainer,
)


@step
def train_ultralytics_model_context(
    model_context: UltralyticsModelContext,
    dataset_collection: DatasetCollection[TBaseDatasetContext],
) -> UltralyticsModelContext:
    """
    Trains an Ultralytics model on the provided dataset collection.

    This function retrieves the active training context and initializes an `UltralyticsModelContextTrainer`.
    It trains the Ultralytics model using the provided dataset collection, applying the hyperparameters and
    augmentation parameters specified in the context. After training, the updated model context is returned.

    Args:
        model_context (UltralyticsModelContext): The context containing the Ultralytics model to be trained.
        dataset_collection (DatasetCollection): The dataset collection to be used for training the model.

    Returns:
        UltralyticsModelContext: The updated model context after training.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_trainer = UltralyticsModelContextTrainer(
        model_context=model_context,
        experiment=context.experiment,
    )

    model_context = model_context_trainer.train_model_context(
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
        augmentation_parameters=context.augmentation_parameters,
    )

    model_context.set_latest_run_dir()
    model_context.set_trained_weights_path()
    if not model_context.trained_weights_path or not os.path.exists(
        model_context.trained_weights_path
    ):
        raise FileNotFoundError(
            f"Trained weights not found at {model_context.trained_weights_path}"
        )
    model_context.save_artifact_to_experiment(
        experiment=context.experiment,
        artifact_name="best-model",
        artifact_path=model_context.trained_weights_path,
    )

    return model_context

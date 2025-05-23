import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import DatasetCollection
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel
from picsellia_cv_engine.frameworks.ultralytics.parameters.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from picsellia_cv_engine.frameworks.ultralytics.parameters.hyper_parameters import (
    UltralyticsHyperParameters,
)
from picsellia_cv_engine.frameworks.ultralytics.services.model.trainer import (
    UltralyticsModelTrainer,
)

from pipelines.yolov8.training.utils.callbacks import UltralyticsSimpleCallbacks


@step
def simple_train_ultralytics_model(
    model: UltralyticsModel,
    dataset_collection: DatasetCollection[TBaseDataset],
) -> UltralyticsModel:
    """
    Trains an Ultralytics model on the provided dataset collection.

    This function retrieves the active training context and initializes an `UltralyticsModelTrainer`.
    It trains the Ultralytics model using the provided dataset collection, applying the hyperparameters and
    augmentation parameters specified in the context. After training, the updated model is returned.

    Args:
        model (UltralyticsModel): The context containing the Ultralytics model to be trained.
        dataset_collection (DatasetCollection): The dataset collection to be used for training the model.

    Returns:
        UltralyticsModel: The updated model after training.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_trainer = UltralyticsModelTrainer(
        model=model,
        experiment=context.experiment,
    )

    model = model_trainer.train_model(
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
        augmentation_parameters=context.augmentation_parameters,
        callbacks=UltralyticsSimpleCallbacks,
    )

    model.set_latest_run_dir()
    model.set_trained_weights_path()
    if not model.trained_weights_path or not os.path.exists(model.trained_weights_path):
        raise FileNotFoundError(
            f"Trained weights not found at {model.trained_weights_path}"
        )
    model.save_artifact_to_experiment(
        experiment=context.experiment,
        artifact_name="best-model",
        artifact_path=model.trained_weights_path,
    )

    return model

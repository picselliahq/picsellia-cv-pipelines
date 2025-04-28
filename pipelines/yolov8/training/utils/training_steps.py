import os
from pathlib import Path

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import DatasetCollection
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.models.ultralytics.model import UltralyticsModel
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.core.parameters.ultralytics.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from picsellia_cv_engine.core.parameters.ultralytics.hyper_parameters import (
    UltralyticsHyperParameters,
)
from picsellia_cv_engine.services.ultralytics.model.callbacks import (
    TBaseTrainer,
    TBaseValidator,
    UltralyticsCallbacks,
)
from picsellia_cv_engine.services.ultralytics.model.trainer import (
    UltralyticsModelTrainer,
)


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
        model=model, experiment=context.experiment, callbacks=UltralyticsSimpleCallbacks
    )

    model = model_trainer.train_model(
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
        augmentation_parameters=context.augmentation_parameters,
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


class UltralyticsSimpleCallbacks(UltralyticsCallbacks):
    def on_train_epoch_end(self, trainer: TBaseTrainer):
        """
        Logs metrics and learning rate at the end of each training epoch.

        Args:
            trainer (TBaseTrainer): The trainer instance containing current training state and metrics.
        """
        for metric_name, loss_value in trainer.label_loss_items(trainer.tloss).items():
            if metric_name.startswith("val") or metric_name.startswith("metrics"):
                self.logger.log_metric(
                    name=metric_name, value=float(loss_value), phase="val"
                )
            else:
                self.logger.log_metric(
                    name=metric_name, value=float(loss_value), phase="train"
                )

        for lr_name, lr_value in trainer.lr.items():
            self.logger.log_metric(name=lr_name, value=float(lr_value), phase="train")

    def on_fit_epoch_end(self, trainer: TBaseTrainer):
        """
        Logs the time and metrics at the end of each epoch.

        Args:
            trainer (TBaseTrainer): The trainer instance containing current training state and metrics.
        """
        self.logger.log_metric(
            name="epoch_time", value=float(trainer.epoch_time), phase="train"
        )

        for metric_name, metric_value in trainer.metrics.items():
            if metric_name.startswith("val") or metric_name.startswith("metrics"):
                self.logger.log_metric(
                    name=metric_name, value=float(metric_value), phase="val"
                )
            else:
                self.logger.log_metric(
                    name=metric_name, value=float(metric_value), phase="train"
                )

    def on_val_end(self, validator: TBaseValidator):
        """
        Logs validation results including validation images at the end of the validation phase.

        Args:
            validator (TBaseValidator): The validator instance containing the validation results.
        """
        val_output_directory = Path(validator.save_dir)

        valid_prefixes = ("val", "P", "R", "F1", "Box", "Mask")
        files = [
            file
            for file in val_output_directory.iterdir()
            if file.stem.startswith(valid_prefixes)
        ]
        existing_files: list[Path] = [
            val_output_directory / file_name for file_name in files
        ]
        for file_path in existing_files:
            self.logger.log_image(
                name=file_path.stem, image_path=str(file_path), phase="val"
            )

from pathlib import Path
from typing import TypeVar

import numpy as np
from picsellia import Experiment
from picsellia_cv_engine.models.steps.model.logging.base_logger import (
    BaseLogger,
    MetricMapping,
)
from ultralytics.engine.trainer import BaseTrainer

TBaseLogger = TypeVar("TBaseLogger", bound=BaseLogger)
TMetricMapping = TypeVar("TMetricMapping", bound=MetricMapping)
TBaseTrainer = TypeVar("TBaseTrainer", bound=BaseTrainer)


class UltralyticsCallbacks:
    """
    A class that provides callback methods for logging metrics, images, and results during the
    training and validation process of an Ultralytics YOLO model.
    """

    def __init__(
        self,
        experiment: Experiment,
        logger: type[TBaseLogger],
        metric_mapping: TMetricMapping,
    ):
        """
        Initializes the callback class with an experiment for logging.

        Args:
            experiment (Experiment): The experiment instance for logging training and validation data.
            logger (Type[TBaseLogger]): The logger class to be used for logging metrics and images.
            metric_mapping (TMetricMapping): The metric mapping class for mapping
        """
        self.logger = logger(experiment=experiment, metric_mapping=metric_mapping)

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

        if trainer.epoch == 0:
            from ultralytics.utils.torch_utils import model_info_for_loggers

            for info_key, info_value in model_info_for_loggers(trainer).items():
                self.logger.log_value(name=info_key, value=info_value)

    def on_val_end(self, validator: TBaseTrainer):
        """
        Logs validation results including validation images at the end of the validation phase.

        Args:
            validator (TBaseTrainer): The validator instance containing validation results.
        """
        validation_images_directory = Path(validator.save_dir)
        image_files = sorted(validation_images_directory.glob("val*.jpg"))
        for image_file in image_files:
            self.logger.log_image(
                name=image_file.stem.replace("val_", ""),
                image_path=str(image_file),
                phase="val",
            )

    def on_train_end(self, trainer: TBaseTrainer):
        """
        Logs the final results, including metrics and visualizations, after training completes.

        Args:
            trainer (TBaseTrainer): The trainer instance containing the final training state.
        """
        model_output_directory = Path(trainer.save_dir)
        visualization_files = [
            "results.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            *(f"{metric}_curve.png" for metric in ("F1", "PR", "P", "R")),
        ]
        existing_files: list[Path] = [
            model_output_directory / file_name
            for file_name in visualization_files
            if (model_output_directory / file_name).exists()
        ]
        for file_path in existing_files:
            self.logger.log_image(
                name=file_path.stem, image_path=str(file_path), phase="val"
            )

        for metric_key, metric_value in trainer.validator.metrics.results_dict.items():
            if isinstance(metric_value, np.float64):
                metric_value = float(metric_value)
            self.logger.log_value(
                name=f"{metric_key}-final", value=metric_value, phase="val"
            )

    def get_callbacks(self):
        """
        Returns a dictionary mapping callback names to the corresponding callback functions.

        Returns:
            dict: A dictionary of callback functions.
        """
        return {
            "on_train_epoch_end": self.on_train_epoch_end,
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_val_end": self.on_val_end,
            "on_train_end": self.on_train_end,
        }

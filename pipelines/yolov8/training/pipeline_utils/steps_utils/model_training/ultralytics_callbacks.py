from pathlib import Path
from typing import TypeVar

from picsellia import Experiment
from picsellia_cv_engine.services.base.model.logging.base_logger import (
    BaseLogger,
    MetricMapping,
)
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator

TBaseLogger = TypeVar("TBaseLogger", bound=BaseLogger)
TMetricMapping = TypeVar("TMetricMapping", bound=MetricMapping)
TBaseTrainer = TypeVar("TBaseTrainer", bound=BaseTrainer)
TBaseValidator = TypeVar("TBaseValidator", bound=BaseValidator)


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

        train_output_directory = Path(trainer.save_dir)
        valid_prefixes = ("train", "labels")
        files = [
            file
            for file in train_output_directory.iterdir()
            if file.stem.startswith(valid_prefixes)
        ]
        existing_files: list[Path] = [
            train_output_directory / file_name for file_name in files
        ]
        for file_path in existing_files:
            self.logger.log_image(
                name=file_path.stem, image_path=str(file_path), phase="train"
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

        desc = validator.get_desc()
        column_names = extract_column_names(desc)

        if hasattr(validator, "metrics") and hasattr(
            validator.metrics, "ap_class_index"
        ):  # Detection or Segmentation
            for i, c in enumerate(validator.metrics.ap_class_index):
                class_name = validator.names[c]
                row = {
                    "Class": class_name,
                    "Images": int(validator.nt_per_image[c]),
                    "Instances": int(validator.nt_per_class[c]),
                }

                metrics = validator.metrics.class_result(i)
                for j, col in enumerate(
                    column_names[3:]
                ):  # Skip Class, Images, Instances
                    row[col] = round(metrics[j], 3)

                # Log one table per class
                self.logger.log_table(
                    name=f"{class_name}-metrics", data=row, phase="val"
                )

        elif hasattr(validator, "metrics") and hasattr(
            validator.metrics, "top1"
        ):  # Classification
            row = {
                "classes": "all",
                "top1_acc": round(validator.metrics.top1, 3),
                "top5_acc": round(validator.metrics.top5, 3),
            }
            self.logger.log_table(name="classification-metrics", data=row, phase="val")

        if (
            hasattr(validator, "confusion_matrix")
            and validator.confusion_matrix is not None
        ):
            matrix = validator.confusion_matrix.matrix
            if hasattr(validator, "metrics") and hasattr(
                validator.metrics, "ap_class_index"
            ):
                labelmap = dict(
                    enumerate(list(validator.names.values()) + ["background"])
                )

            else:
                labelmap = dict(enumerate(validator.names.values()))
            self.logger.log_confusion_matrix(
                name="confusion_matrix", labelmap=labelmap, matrix=matrix, phase="val"
            )

    def on_train_end(self, trainer: TBaseTrainer):
        """
        Logs the final results, including metrics and visualizations, after training completes.

        Args:
            trainer (TBaseTrainer): The trainer instance containing the final training state.
        """
        # for metric_key, metric_value in trainer.validator.metrics.results_dict.items():
        #     if isinstance(metric_value, np.float64):
        #         metric_value = float(metric_value)
        #     self.logger.log_value(
        #         name=f"{metric_key}-final", value=metric_value, phase="val"
        #     )

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


def extract_column_names(desc: str) -> list[str]:
    """
    Extracts column names from a YOLO-style desc string, preserving prefixes like 'Box' or 'Mask'.

    Example:
    Input:  "%22s" + "%11s" * 10 => "Class Images Instances Box(P R mAP50 mAP50-95) Mask(P R mAP50 mAP50-95)"
    Output: ['Class', 'Images', 'Instances', 'Box(P)', 'Box(R)', 'Box(mAP50)', 'Box(mAP50-95)', 'Mask(P)', 'Mask(R)', ...]
    """
    tokens = desc.replace(")", ") ").split()
    final_cols = []

    current_prefix = None
    for token in tokens:
        if "(" in token:
            current_prefix = token.split("(")[0]
            metric = token[token.find("(") + 1 :].rstrip(")")
            final_cols.append(f"{current_prefix}({metric})")
        elif current_prefix:
            final_cols.append(f"{current_prefix}({token.rstrip(')')})")
        else:
            final_cols.append(token)

    return final_cols

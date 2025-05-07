from pathlib import Path

from picsellia_cv_engine.frameworks.ultralytics.services.model.callbacks import (
    TBaseTrainer,
    TBaseValidator,
    UltralyticsCallbacks,
)


class UltralyticsSimpleCallbacks(UltralyticsCallbacks):
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

import logging
import os
import shutil

from picsellia_cv_engine.models.steps.model.export.model_exporter import ModelExporter
from ultralytics import YOLO

from pipelines.yolov8.training.pipeline_utils.model.ultralytics_model import (
    UltralyticsModel,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)

logger = logging.getLogger(__name__)


class UltralyticsModelExporter(ModelExporter):
    """
    Exporter class for Ultralytics models.

    This class handles the export of models trained with the Ultralytics framework. It supports exporting the model
    to a specified format (e.g., ONNX) and moving the resulting file to a designated destination for deployment or further use.

    Attributes:
        model (UltralyticsModel): The Ultralytics model to be exported.
    """

    def __init__(self, model: UltralyticsModel):
        """
        Initializes an instance of UltralyticsModelExporter.

        Args:
            model (UltralyticsModel): The model containing details about the model and paths.
        """
        super().__init__(model=model)
        self.model: UltralyticsModel = model

    def export_model(
        self,
        exported_model_destination_path: str,
        export_format: str,
        hyperparameters: UltralyticsHyperParameters,
    ) -> None:
        """
        Exports the Ultralytics model by converting it to the specified format (e.g., ONNX) and
        moves the resulting file to the designated destination path.

        Args:
            exported_model_destination_path (str): The path to save the exported model weights.
            export_format (str): The format in which to export the model (e.g., ONNX).
            hyperparameters (UltralyticsHyperParameters): The hyperparameters guiding the export process.

        Raises:
            ValueError: If no results folder or ONNX file is found during the export process.
        """
        self._export_model(export_format=export_format, hyperparameters=hyperparameters)

        onnx_file_path = self._find_exported_onnx_file()

        self._move_onnx_to_destination_path(
            onnx_file_path=onnx_file_path,
            exported_model_destination_path=exported_model_destination_path,
        )

    def _export_model(
        self, export_format: str, hyperparameters: UltralyticsHyperParameters
    ) -> None:
        """
        Exports the loaded model in the specified format (e.g., ONNX) to the model's inference path.

        Args:
            export_format (str): The format to export the model in (e.g., ONNX).
            hyperparameters (UltralyticsHyperParameters): Hyperparameters specifying the image size, batch size, etc.
        """
        loaded_model: YOLO = self.model.loaded_model
        loaded_model.export(
            format=export_format,
            imgsz=hyperparameters.image_size,
            dynamic=False,
            batch=1,
            opset=18,  # ONNX opset version compatible with IR 8
        )

    def _find_exported_onnx_file(self) -> str:
        """
        Searches for the ONNX file in the weights directory of the Ultralytics results folder.

        Returns:
            str: The full path to the ONNX file.

        Raises:
            ValueError: If no ONNX file is found in the weights directory.
        """
        if not self.model.latest_run_dir:
            raise ValueError("The latest run directory is not set.")
        ultralytics_weights_dir = os.path.join(self.model.latest_run_dir, "weights")
        onnx_files = [
            f for f in os.listdir(ultralytics_weights_dir) if f.endswith(".onnx")
        ]
        if not onnx_files:
            raise ValueError("No ONNX file found")
        return os.path.join(ultralytics_weights_dir, onnx_files[0])

    def _move_onnx_to_destination_path(
        self, onnx_file_path: str, exported_model_destination_path: str
    ) -> None:
        """
        Moves the ONNX file from its current location to the specified destination path.

        If a file already exists at the destination, it will be overwritten.

        Args:
            onnx_file_path (str): The full path to the ONNX file.
            exported_model_destination_path (str): The destination path to move the ONNX file.
        """
        logger.info(f"Moving ONNX file to {exported_model_destination_path}...")

        if os.path.exists(
            os.path.join(
                exported_model_destination_path, os.path.basename(onnx_file_path)
            )
        ):
            logger.warning(
                f"File already exists at destination. Removing: {os.path.join(exported_model_destination_path, os.path.basename(onnx_file_path))}"
            )
            os.remove(
                os.path.join(
                    exported_model_destination_path, os.path.basename(onnx_file_path)
                )
            )

        shutil.move(onnx_file_path, exported_model_destination_path)
        logger.info("Move completed successfully.")

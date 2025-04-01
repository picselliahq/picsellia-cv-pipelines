import logging

from picsellia import Experiment

from pipelines.paddle_ocr.pipeline_utils.model.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from pipelines.paddle_ocr.pipeline_utils.steps_utils.model_export.paddle_ocr_model_context_exporter import (
    PaddleOCRModelExporter,
)

logger = logging.getLogger(__name__)


class PaddleOCRModelCollectionExporter:
    """
    Handles the export of a collection of PaddleOCR models (bounding box and text recognition models).

    This class exports the trained models in the provided PaddleOCRModelCollection, saving them
    in the specified format and to the Picsellia experiment.

    Attributes:
        model_collection (PaddleOCRModelCollection): The collection of models to export, containing the bounding box and text models.
        experiment (Experiment): The Picsellia experiment where the models will be saved.
        bbox_model_exporter (PaddleOCRModelExporter): Exporter for the bounding box model.
        text_model_exporter (PaddleOCRModelExporter): Exporter for the text recognition model.
    """

    def __init__(self, model_collection: PaddleOCRModelCollection):
        """
        Initializes the PaddleOCRModelCollectionExporter with the given model collection.

        Args:
            model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models to export.
        """
        self.model_collection = model_collection
        self.bbox_model_exporter = PaddleOCRModelExporter(
            model=self.model_collection.bbox_model
        )
        self.text_model_exporter = PaddleOCRModelExporter(
            model=self.model_collection.text_model
        )

    def export_model_collection(self, export_format: str) -> PaddleOCRModelCollection:
        """
        Exports the trained models in the model collection to the specified format.

        This method handles the export process for both the bounding box and text recognition models.
        If the models' export directories are not found, an error is raised. After successful export,
        the updated model collection is returned.

        Args:
            export_format (str): The format in which the models will be exported (e.g., 'onnx', 'tensorflow').

        Returns:
            PaddleOCRModelCollection: The updated model collection with exported models.

        Raises:
            ValueError: If the exported weights directory is not found in the model collection.
        """
        if (
            not self.model_collection.bbox_model.exported_weights_dir
            or not self.model_collection.text_model.exported_weights_dir
        ):
            raise ValueError("No exported weights directory found in model collection")

        logger.info("Exporting bounding box model...")
        self.bbox_model_exporter.export_model(
            exported_model_destination_path=self.model_collection.bbox_model.exported_weights_dir,
            export_format=export_format,
            hyperparameters=None,
        )

        logger.info("Exporting text recognition model...")
        self.text_model_exporter.export_model(
            exported_model_destination_path=self.model_collection.text_model.exported_weights_dir,
            export_format=export_format,
            hyperparameters=None,
        )
        return self.model_collection

    def save_model_collection(self, experiment: Experiment) -> None:
        """
        Saves the exported models from the model collection to the Picsellia experiment.

        This method uploads the exported bounding box and text recognition models to the associated experiment.
        If the models' export directories are not found, an error is raised.

        Raises:
            ValueError: If the exported weights directory is not found in the model collection.
        """
        if (
            not self.model_collection.bbox_model.exported_weights_dir
            or not self.model_collection.text_model.exported_weights_dir
        ):
            raise ValueError("No exported weights directory found in model collection")

        logger.info("Saving bounding box model to experiment...")
        self.bbox_model_exporter.save_model_to_experiment(
            experiment=experiment,
            exported_weights_dir=self.model_collection.bbox_model.exported_weights_dir,
            exported_weights_name="bbox-model-latest",
        )

        logger.info("Saving text recognition model to experiment...")
        self.text_model_exporter.save_model_to_experiment(
            experiment=experiment,
            exported_weights_dir=self.model_collection.text_model.exported_weights_dir,
            exported_weights_name="text-model-latest",
        )

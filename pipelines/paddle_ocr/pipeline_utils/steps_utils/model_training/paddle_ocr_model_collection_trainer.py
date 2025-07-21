from picsellia import Experiment

from paddle_ocr.pipeline_utils.model.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from paddle_ocr.pipeline_utils.steps_utils.model_training.paddle_ocr_model_context_trainer import (
    PaddleOCRModelTrainer,
)


class PaddleOCRModelCollectionTrainer:
    """
    Trains a collection of PaddleOCR models, including both bounding box detection and text recognition models.

    This class manages the training process for multiple models in the `PaddleOCRModelCollection` and logs
    the progress to the specified Picsellia experiment.

    Attributes:
        model_collection (PaddleOCRModelCollection): The collection of models (bounding box and text recognition models).
        experiment (Experiment): The Picsellia experiment where the training logs and metrics are recorded.
        last_logged_epoch (Union[int, None]): Tracks the last epoch that was logged for both models.
    """

    def __init__(
        self, model_collection: PaddleOCRModelCollection, experiment: Experiment
    ):
        """
        Initializes the `PaddleOCRModelCollectionTrainer` with a model collection and experiment.

        Args:
            model_collection (PaddleOCRModelCollection): The collection of models to be trained.
            experiment (Experiment): The Picsellia experiment where logs and metrics will be recorded.
        """
        self.model_collection = model_collection
        self.experiment = experiment
        self.last_logged_epoch: int | None = None  # Last epoch that was logged

    def train_model_collection(
        self, bbox_epochs: int, text_epochs: int
    ) -> PaddleOCRModelCollection:
        """
        Trains the models in the collection based on the number of epochs specified for each model.

        This method trains both the bounding box detection and text recognition models if the number
        of epochs is greater than 0 for each. If the number of epochs for a model is set to 0, training
        for that model is skipped.

        Args:
            bbox_epochs (int): The number of epochs to train the bounding box detection model.
            text_epochs (int): The number of epochs to train the text recognition model.

        Returns:
            PaddleOCRModelCollection: The updated model collection after training.

        Raises:
            ValueError: If no epochs are provided for both models.
        """
        if bbox_epochs > 0:
            print("Starting training for bounding box model...")
            model_trainer = PaddleOCRModelTrainer(
                model=self.model_collection.bbox_model,
                experiment=self.experiment,
            )
            model_trainer.train_model()
        else:
            print("Skipping training for bounding box model...")

        if text_epochs > 0:
            print("Starting training for text recognition model...")
            model_trainer = PaddleOCRModelTrainer(
                model=self.model_collection.text_model,
                experiment=self.experiment,
            )
            model_trainer.train_model()
        else:
            print("Skipping training for text recognition model...")

        return self.model_collection

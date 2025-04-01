# type: ignore
from picsellia_cv_engine.core.contexts import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.paddle_ocr.pipeline_utils.model.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from pipelines.paddle_ocr.pipeline_utils.parameters.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from pipelines.paddle_ocr.pipeline_utils.parameters.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from pipelines.paddle_ocr.pipeline_utils.steps_utils.model_training.paddle_ocr_model_collection_trainer import (
    PaddleOCRModelCollectionTrainer,
)


@step
def train_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
) -> PaddleOCRModelCollection:
    """
    Trains a PaddleOCR model collection based on the provided hyperparameters.

    This function retrieves the active training context from the pipeline and initializes a
    `PaddleOCRModelCollectionTrainer`. It then trains the model collection, including both the
    bounding box detection and text recognition models, for the number of epochs specified in the
    hyperparameters. After training, the updated model collection is returned.

    Args:
        model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models (bounding box and text recognition) to be trained.

    Returns:
        PaddleOCRModelCollection: The trained model collection.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_trainer = PaddleOCRModelCollectionTrainer(
        model_collection=model_collection, experiment=context.experiment
    )

    model_collection = model_trainer.train_model_collection(
        bbox_epochs=context.hyperparameters.bbox_epochs,
        text_epochs=context.hyperparameters.text_epochs,
    )

    return model_collection

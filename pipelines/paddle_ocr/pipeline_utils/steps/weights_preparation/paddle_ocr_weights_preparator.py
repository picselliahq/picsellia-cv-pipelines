# type: ignore
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.data.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)

from pipelines.paddle_ocr.pipeline_utils.dataset.paddle_ocr_dataset import (
    PaddleOCRDataset,
)
from pipelines.paddle_ocr.pipeline_utils.model.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from pipelines.paddle_ocr.pipeline_utils.parameters.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from pipelines.paddle_ocr.pipeline_utils.parameters.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from pipelines.paddle_ocr.pipeline_utils.steps_utils.weights_preparation.paddle_ocr_model_collection_preparator import (
    PaddleOCRModelCollectionPreparator,
)


@step
def prepare_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
    dataset_collection: DatasetCollection[PaddleOCRDataset],
) -> PaddleOCRModelCollection:
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()
    model_collection_preparator = PaddleOCRModelCollectionPreparator(
        model_collection=model_collection,
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
    )
    model_collection = model_collection_preparator.prepare()
    return model_collection

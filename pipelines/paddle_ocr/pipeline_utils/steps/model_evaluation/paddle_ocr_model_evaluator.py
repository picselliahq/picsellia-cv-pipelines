# type: ignore
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.data.dataset.base_dataset import (
    TBaseDataset,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.models.steps.model.evaluator.model_evaluator import (
    ModelEvaluator,
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
from pipelines.paddle_ocr.pipeline_utils.steps_utils.model_prediction.paddle_ocr_model_collection_predictor import (
    PaddleOCRModelCollectionPredictor,
)


@step
def evaluate_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
    dataset: TBaseDataset,
) -> None:
    """
    Evaluates a PaddleOCR model collection on a given dataset.

    This function retrieves the active training context from the pipeline, performs inference using
    the provided PaddleOCR model collection on the dataset, and evaluates the predictions. It processes
    the dataset in batches, runs inference, and then logs the evaluation results to the experiment.

    Args:
        model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models to be evaluated.
        dataset (TDataset): The dataset containing the data for evaluation.

    Returns:
        None: The function performs evaluation and logs the results but does not return any value.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_collection_predictor = PaddleOCRModelCollectionPredictor(
        model_collection=model_collection,
    )
    image_paths = model_collection_predictor.pre_process_dataset(dataset=dataset)
    image_batches = model_collection_predictor.prepare_batches(
        image_paths=image_paths,
        batch_size=min(
            context.hyperparameters.bbox_batch_size,
            context.hyperparameters.text_batch_size,
        ),
    )
    batch_results = model_collection_predictor.run_inference_on_batches(
        image_batches=image_batches
    )
    picsellia_ocr_predictions = model_collection_predictor.post_process_batches(
        image_batches=image_batches,
        batch_results=batch_results,
        dataset=dataset,
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment,
        inference_type=model_collection.bbox_model.model_version.type,
    )
    model_evaluator.evaluate(picsellia_predictions=picsellia_ocr_predictions)

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.models.steps.model_evaluation.model_evaluator import (
    ModelEvaluator,
)

from pipelines.yolov8_classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)
from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8_classification.pipeline_utils.steps_utils.model_prediction.classification_model_context_predictor import (
    UltralyticsClassificationModelContextPredictor,
)


@step
def evaluate_ultralytics_model_context(
    model_context: UltralyticsModelContext,
    dataset_context: TBaseDatasetContext,
) -> None:
    """
    Evaluates an Ultralytics classification model on a given dataset.

    This function retrieves the active training context from the pipeline, performs inference using
    the provided Ultralytics classification model on the dataset, and evaluates the predictions. It processes
    the dataset in batches, runs inference, and then logs the evaluation results to the experiment.

    Args:
        model_context (ModelContext): The Ultralytics model context to be evaluated.
        dataset_context (TDatasetContext): The dataset context containing the data for evaluation.

    Returns:
        None: The function performs evaluation and logs the results to the experiment but does not return any value.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_predictor = UltralyticsClassificationModelContextPredictor(
        model_context=model_context
    )
    image_paths = model_context_predictor.pre_process_dataset_context(
        dataset_context=dataset_context
    )
    image_batches = model_context_predictor.prepare_batches(
        image_paths=image_paths, batch_size=context.hyperparameters.batch_size
    )
    batch_results = model_context_predictor.run_inference_on_batches(
        image_batches=image_batches
    )
    picsellia_classifications_predictions = (
        model_context_predictor.post_process_batches(
            image_batches=image_batches,
            batch_results=batch_results,
            dataset_context=dataset_context,
        )
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment, inference_type=model_context.model_version.type
    )
    model_evaluator.evaluate(
        picsellia_predictions=picsellia_classifications_predictions
    )

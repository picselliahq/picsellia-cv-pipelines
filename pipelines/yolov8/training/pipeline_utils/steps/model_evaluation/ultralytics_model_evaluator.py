import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.services.base.model.evaluator.model_evaluator import (
    ModelEvaluator,
)

from pipelines.yolov8.training.classification.pipeline_utils.steps_utils.model_prediction.classification_model_context_predictor import (
    UltralyticsClassificationModelPredictor,
)
from pipelines.yolov8.training.object_detection.pipeline_utils.steps_utils.model_prediction.object_detection_model_context_predictor import (
    UltralyticsDetectionModelPredictor,
)
from pipelines.yolov8.training.pipeline_utils.model.ultralytics_model import (
    UltralyticsModel,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8.training.segmentation.pipeline_utils.steps_utils.model_prediction.segmentation_model_context_predictor import (
    UltralyticsSegmentationModelPredictor,
)


@step
def evaluate_ultralytics_model(
    model: UltralyticsModel,
    dataset: TBaseDataset,
) -> None:
    """
    Evaluates an Ultralytics classification model on a given dataset.

    This function retrieves the active training context from the pipeline, performs inference using
    the provided Ultralytics classification model on the dataset, and evaluates the predictions. It processes
    the dataset in batches, runs inference, and then logs the evaluation results to the experiment.

    Args:
        model (Model): The Ultralytics model to be evaluated.
        dataset (TDataset): The dataset containing the data for evaluation.

    Returns:
        None: The function performs evaluation and logs the results to the experiment but does not return any value.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    if model.loaded_model.task == "classify":
        model_predictor = UltralyticsClassificationModelPredictor(model=model)
    elif model.loaded_model.task == "detect":
        model_predictor = UltralyticsDetectionModelPredictor(model=model)
    elif model.loaded_model.task == "segment":
        model_predictor = UltralyticsSegmentationModelPredictor(model=model)
    else:
        raise ValueError(f"Model task {model.loaded_model.task} not supported")

    image_paths = model_predictor.pre_process_dataset(dataset=dataset)
    image_batches = model_predictor.prepare_batches(
        image_paths=image_paths, batch_size=context.hyperparameters.batch_size
    )
    batch_results = model_predictor.run_inference_on_batches(
        image_batches=image_batches
    )
    picsellia_predictions = model_predictor.post_process_batches(
        image_batches=image_batches,
        batch_results=batch_results,
        dataset=dataset,
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment, inference_type=model.model_version.type
    )
    model_evaluator.evaluate(picsellia_predictions=picsellia_predictions)

    if model.loaded_model.task == "classify":
        model_evaluator.compute_classification_metrics(
            assets=dataset.assets,
            output_dir=os.path.join(model.results_dir, "evaluation"),
        )
    elif model.loaded_model.task == "detect" or model.loaded_model.task == "segment":
        model_evaluator.compute_coco_metrics(
            assets=dataset.assets,
            output_dir=os.path.join(model.results_dir, "evaluation"),
        )

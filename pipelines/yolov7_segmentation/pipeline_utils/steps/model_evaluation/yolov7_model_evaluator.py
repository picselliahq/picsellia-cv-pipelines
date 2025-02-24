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

from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model_context import (
    Yolov7ModelContext,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps_utils.model_prediction.segmentation_model_context_predictor import (
    Yolov7SegmentationModelContextPredictor,
)


@step
def yolov7_model_context_evaluator(
    model_context: Yolov7ModelContext,
    dataset_context: TBaseDatasetContext,
) -> None:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_predictor = Yolov7SegmentationModelContextPredictor(
        model_context=model_context
    )
    image_paths = model_context_predictor.pre_process_dataset_context(
        dataset_context=dataset_context
    )
    label_path_to_mask_paths = model_context_predictor.run_inference(
        image_paths=image_paths,
        hyperparameters=context.hyperparameters,
    )
    picsellia_polygons_predictions = model_context_predictor.post_process(
        label_path_to_mask_paths=label_path_to_mask_paths,
        dataset_context=dataset_context,
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment, inference_type=model_context.model_version.type
    )
    model_evaluator.evaluate(picsellia_predictions=picsellia_polygons_predictions)

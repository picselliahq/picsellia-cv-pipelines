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

from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model import (
    Yolov7Model,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps_utils.model_prediction.segmentation_model_predictor import (
    Yolov7SegmentationModelPredictor,
)


@step
def yolov7_model_evaluator(
    model: Yolov7Model,
    dataset: TBaseDataset,
) -> None:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_predictor = Yolov7SegmentationModelPredictor(model=model)
    image_paths = model_predictor.pre_process_dataset(dataset=dataset)
    label_path_to_mask_paths = model_predictor.run_inference(
        image_paths=image_paths,
        hyperparameters=context.hyperparameters,
    )
    picsellia_polygons_predictions = model_predictor.post_process(
        label_path_to_mask_paths=label_path_to_mask_paths,
        dataset=dataset,
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment, inference_type=model.model_version.type
    )
    model_evaluator.evaluate(picsellia_predictions=picsellia_polygons_predictions)

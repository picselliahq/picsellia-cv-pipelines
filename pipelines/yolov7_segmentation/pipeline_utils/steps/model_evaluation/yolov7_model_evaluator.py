from picsellia_cv_engine.core.contexts import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.data import (
    TBaseDataset,
)
from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.core.services.model.evaluator.model_evaluator import (
    ModelEvaluator,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.steps.base.model.evaluator import evaluate_model_impl

from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model_context import (
    Yolov7Model,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps_utils.model_prediction.segmentation_model_context_predictor import (
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

    evaluate_model_impl(
        context=context,
        picsellia_predictions=picsellia_polygons_predictions,
        inference_type=model.model_version.type,
        assets=dataset.assets,
        output_dir=context.experiment.get_logs_dir(),
    )

from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.steps.data_validation.yolo_segmentation_dataset_collection_validator import (
    yolo_segmentation_dataset_collection_validator,
)

from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps.data_extraction.yolov7_data_extractor import (
    yolov7_dataset_collection_extractor,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps.data_preparation.yolov7_data_preparator import (
    yolov7_dataset_collection_preparator,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps.model_evaluation.yolov7_model_evaluator import (
    yolov7_model_evaluator,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps.model_training.yolov7_trainer import (
    yolov7_model_trainer,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps.weights_extraction.yolov7_weights_extractor import (
    yolov7_model_extractor,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps.weights_preparation.yolov7_weights_preparator import (
    yolov7_model_preparator,
)


def get_context() -> PicselliaTrainingContext[
    Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
]:
    return PicselliaTrainingContext(
        hyperparameters_cls=Yolov7HyperParameters,
        augmentation_parameters_cls=Yolov7AugmentationParameters,
        export_parameters_cls=ExportParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov7_segmentation_training_pipeline():
    dataset_collection = yolov7_dataset_collection_extractor()
    dataset_collection = yolov7_dataset_collection_preparator(
        dataset_collection=dataset_collection
    )
    yolo_segmentation_dataset_collection_validator(
        dataset_collection=dataset_collection, fix_annotation=True
    )

    model = yolov7_model_extractor(
        pretrained_weights_name="pretrained-weights",
        config_name="config",
        hyperparameters_name="hyperparameters",
    )
    model = yolov7_model_preparator(model=model)
    model = yolov7_model_trainer(model=model, dataset_collection=dataset_collection)

    yolov7_model_evaluator(model=model, dataset=dataset_collection["test"])


if __name__ == "__main__":
    yolov7_segmentation_training_pipeline()

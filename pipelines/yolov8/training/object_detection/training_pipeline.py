from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.steps.dataset.loader import load_yolo_datasets
from picsellia_cv_engine.steps.dataset.validator import validate_dataset

from pipelines.yolov8.training.classification.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8.training.classification.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps.model_evaluation.ultralytics_model_evaluator import (
    evaluate_ultralytics_model_context,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps.model_export.ultralytics_model_exporter import (
    export_ultralytics_model_context,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps.model_loading.ultralytics_model_context_loader import (
    load_ultralytics_model_context,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps.model_training.ultralytics_trainer import (
    train_ultralytics_model_context,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_ultralytics_model_context,
)
from pipelines.yolov8.training.object_detection.pipeline_utils.steps.data_preparation.ultralytics_data_preparator import (
    prepare_ultralytics_dataset_collection,
)


def get_context() -> PicselliaTrainingContext[
    UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
]:
    return PicselliaTrainingContext(
        hyperparameters_cls=UltralyticsHyperParameters,
        augmentation_parameters_cls=UltralyticsAugmentationParameters,
        export_parameters_cls=ExportParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_object_detection_training_pipeline():
    dataset_collection = load_yolo_datasets()
    prepare_ultralytics_dataset_collection(dataset_collection=dataset_collection)
    validate_dataset(dataset=dataset_collection, fix_annotation=True)

    model_context = get_ultralytics_model_context(
        pretrained_weights_name="pretrained-weights"
    )
    load_ultralytics_model_context(
        model_context=model_context,
        weights_path_to_load=model_context.pretrained_weights_path,
    )
    train_ultralytics_model_context(
        model_context=model_context, dataset_collection=dataset_collection
    )

    export_ultralytics_model_context(model_context=model_context)
    evaluate_ultralytics_model_context(
        model_context=model_context, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    yolov8_object_detection_training_pipeline()

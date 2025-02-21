from src.picsellia_cv_engine import pipeline
from src.picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)
from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.picsellia_cv_engine.steps.data_extraction.coco_data_extractor import (
    get_coco_dataset_collection,
)
from pipelines.yolov8_classification.pipeline_utils.steps.data_preparation.ultralytics_classification_data_preparator import (
    prepare_ultralytics_classification_dataset_collection,
)
from src.picsellia_cv_engine.steps.data_validation.coco_classification_dataset_collection_validator import (
    validate_coco_classification_dataset_collection,
)
from pipelines.yolov8_classification.pipeline_utils.steps.model_evaluation.ultralytics_model_evaluator import (
    evaluate_ultralytics_model_context,
)
from pipelines.yolov8_classification.pipeline_utils.steps.model_export.ultralytics_model_exporter import (
    export_ultralytics_model_context,
)
from pipelines.yolov8_classification.pipeline_utils.steps.model_loading.ultralytics_model_context_loader import (
    load_ultralytics_model_context,
)
from pipelines.yolov8_classification.pipeline_utils.steps.model_training.ultralytics_trainer import (
    train_ultralytics_model_context,
)

from pipelines.yolov8_classification.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_ultralytics_model_context,
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
def yolov8_classification_training_pipeline():
    dataset_collection = get_coco_dataset_collection()
    prepare_ultralytics_classification_dataset_collection(
        dataset_collection=dataset_collection
    )
    validate_coco_classification_dataset_collection(
        dataset_collection=dataset_collection
    )

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
    yolov8_classification_training_pipeline()

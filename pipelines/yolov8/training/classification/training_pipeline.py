from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.validator import validate_dataset

from pipelines.yolov8.training.classification.pipeline_utils.steps.data_preparation.ultralytics_classification_data_preparator import (
    prepare_ultralytics_classification_dataset_collection,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8.training.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8.training.pipeline_utils.steps.model_evaluation.ultralytics_model_evaluator import (
    evaluate_ultralytics_model,
)
from pipelines.yolov8.training.pipeline_utils.steps.model_export.ultralytics_model_exporter import (
    export_ultralytics_model,
)
from pipelines.yolov8.training.pipeline_utils.steps.model_loading.ultralytics_model_loader import (
    load_ultralytics_model,
)
from pipelines.yolov8.training.pipeline_utils.steps.model_training.ultralytics_trainer import (
    train_ultralytics_model,
)
from pipelines.yolov8.training.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_ultralytics_model,
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
    dataset_collection = load_coco_datasets()
    prepare_ultralytics_classification_dataset_collection(
        dataset_collection=dataset_collection
    )
    validate_dataset(dataset=dataset_collection, fix_annotation=True)

    model = get_ultralytics_model(pretrained_weights_name="pretrained-weights")
    load_ultralytics_model(
        model=model,
        weights_path_to_load=model.pretrained_weights_path,
    )
    train_ultralytics_model(model=model, dataset_collection=dataset_collection)

    export_ultralytics_model(model=model)
    evaluate_ultralytics_model(model=model, dataset=dataset_collection["test"])


if __name__ == "__main__":
    yolov8_classification_training_pipeline()

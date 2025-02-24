from argparse import ArgumentParser

from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.contexts.training.local_picsellia_training_context import (
    LocalPicselliaTrainingContext,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.steps.data_extraction.coco_data_extractor import (
    get_coco_dataset_collection,
)
from picsellia_cv_engine.steps.data_validation.coco_classification_dataset_collection_validator import (
    validate_coco_classification_dataset_collection,
)
from picsellia_cv_engine.steps.weights_extraction.training_weights_extractor import (
    get_training_model_context,
)

from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8_classification.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8_classification.pipeline_utils.steps.data_preparation.ultralytics_classification_data_preparator import (
    prepare_ultralytics_classification_dataset_collection,
)
from pipelines.yolov8_classification.pipeline_utils.steps.model_evaluation.ultralytics_model_evaluator import (
    evaluate_ultralytics_model_context,
)
from pipelines.yolov8_classification.pipeline_utils.steps.model_loading.ultralytics_model_context_loader import (
    load_ultralytics_model_context,
)

parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--experiment_id", type=str)

args = parser.parse_args()


def get_context() -> LocalPicselliaTrainingContext:
    return LocalPicselliaTrainingContext(
        api_token=args.api_token,
        organization_id=args.organization_id,
        experiment_id=args.experiment_id,
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
    dataset_collection = prepare_ultralytics_classification_dataset_collection(
        dataset_collection=dataset_collection
    )
    validate_coco_classification_dataset_collection(
        dataset_collection=dataset_collection
    )

    model_context = get_training_model_context(
        pretrained_weights_name="pretrained-weights"
    )
    model_context = load_ultralytics_model_context(
        model_context=model_context,
        weights_path_to_load=model_context.pretrained_weights_path,
    )

    evaluate_ultralytics_model_context(
        model_context=model_context, dataset_context=dataset_collection["test"]
    )


if __name__ == "__main__":
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    yolov8_classification_training_pipeline()

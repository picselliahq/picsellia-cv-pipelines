from argparse import ArgumentParser

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.contexts import LocalTrainingContext
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.core.parameters.ultralytics.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from picsellia_cv_engine.core.parameters.ultralytics.hyper_parameters import (
    UltralyticsHyperParameters,
)
from picsellia_cv_engine.steps.ultralytics.dataset.preparator import (
    prepare_ultralytics_dataset,
)
from picsellia_cv_engine.steps.ultralytics.model.evaluator import (
    evaluate_ultralytics_model,
)
from picsellia_cv_engine.steps.ultralytics.model.exporter import (
    export_ultralytics_model,
)
from picsellia_cv_engine.steps.ultralytics.model.loader import load_ultralytics_model
from picsellia_cv_engine.steps.ultralytics.model.trainer import train_ultralytics_model

parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--experiment_id", type=str)

args = parser.parse_args()


def get_context() -> LocalTrainingContext:
    return LocalTrainingContext(
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
def yolov8_training_pipeline():
    dataset_collection = prepare_ultralytics_dataset()

    model = load_ultralytics_model(pretrained_weights_name="pretrained-weights")

    train_ultralytics_model(model=model, dataset_collection=dataset_collection)

    export_ultralytics_model(model=model)

    evaluate_ultralytics_model(model=model, dataset=dataset_collection["test"])


if __name__ == "__main__":
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    yolov8_training_pipeline()

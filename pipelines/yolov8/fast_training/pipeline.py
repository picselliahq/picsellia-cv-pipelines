import argparse

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.core.services.context.unified_context import (
    create_training_context_from_config,
)
from picsellia_cv_engine.frameworks.ultralytics.parameters.augmentation_parameters import (
    UltralyticsAugmentationParameters,
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
from picsellia_cv_engine.steps.ultralytics.model.loader import (
    load_ultralytics_model,
)
from picsellia_cv_engine.steps.ultralytics.model.trainer import (
    train_ultralytics_model,
)
from utils.parameters import (
    UltralyticsHyperParameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_training_context_from_config(
    hyperparameters_cls=UltralyticsHyperParameters,
    augmentation_parameters_cls=UltralyticsAugmentationParameters,
    export_parameters_cls=ExportParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
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

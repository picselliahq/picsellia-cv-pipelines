import argparse

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.core.services.context.unified_context import create_training_context_from_config
from picsellia_cv_engine.frameworks.ultralytics.parameters.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.ultralytics.model.loader import load_ultralytics_model

from steps import train
from utils.parameters import TrainingHyperParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_training_context_from_config(
    hyperparameters_cls=TrainingHyperParameters,
    augmentation_parameters_cls=UltralyticsAugmentationParameters,
    export_parameters_cls=ExportParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)

@pipeline(context=context, log_folder_path="logs/", remove_logs_on_completion=False)
def YOLOv8_OBB_pipeline():
    picsellia_datasets = load_coco_datasets()
    picsellia_model = load_ultralytics_model(pretrained_weights_name="pretrained-weights")
    train(picsellia_model=picsellia_model, picsellia_datasets=picsellia_datasets)


if __name__ == "__main__":
    YOLOv8_OBB_pipeline()

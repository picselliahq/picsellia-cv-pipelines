import argparse

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
)
from picsellia_cv_engine.core.services.context.unified_context import (
    create_training_context_from_config,
)
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.model.builder import build_model
from picsellia_cv_engine.steps.clip.model.evaluator import evaluate
from picsellia_cv_engine.steps.clip.model.trainer import train
from utils.parameters import TrainingHyperParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_training_context_from_config(
    hyperparameters_cls=TrainingHyperParameters,
    augmentation_parameters_cls=AugmentationParameters,
    export_parameters_cls=ExportParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(context=context, log_folder_path="logs/", remove_logs_on_completion=False)
def training_pipeline():
    picsellia_datasets = load_coco_datasets()
    picsellia_model = build_model(
        model_cls=CLIPModel, pretrained_weights_name="pretrained-weights"
    )
    train(model=picsellia_model, dataset_collection=picsellia_datasets)
    evaluate(model=picsellia_model, dataset=picsellia_datasets["test"])


if __name__ == "__main__":
    training_pipeline()

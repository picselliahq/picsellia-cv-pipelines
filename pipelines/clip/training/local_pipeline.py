import argparse

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
)
from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_training_context,
)
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.model.builder import build_model
from picsellia_cv_engine.steps.clip.model.evaluator import evaluate
from picsellia_cv_engine.steps.clip.model.trainer import train
from utils.parameters import TrainingHyperParameters

parser = argparse.ArgumentParser()
parser.add_argument("--api_token", type=str, required=True)
parser.add_argument("--organization_name", type=str, required=True)
parser.add_argument("--experiment_id", type=str, required=True)
parser.add_argument("--working_dir", type=str, required=True)
args = parser.parse_args()

context = create_local_training_context(
    hyperparameters_cls=TrainingHyperParameters,
    augmentation_parameters_cls=AugmentationParameters,
    export_parameters_cls=ExportParameters,
    api_token=args.api_token,
    organization_name=args.organization_name,
    experiment_id=args.experiment_id,
    working_dir=args.working_dir,
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

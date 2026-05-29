import argparse

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
)
from picsellia_cv_engine.core.services.context.unified_context import (
    create_training_context_from_config,
)
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets

from steps import evaluate_sam3_model, load_sam3_model, train_sam3_model
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
def SAM_3_pipeline():
    datasets = load_coco_datasets()
    model, processor = load_sam3_model()
    train_sam3_model(model=model, processor=processor, datasets=datasets)
    evaluate_sam3_model(model=model, processor=processor, datasets=datasets)


if __name__ == "__main__":
    SAM_3_pipeline()

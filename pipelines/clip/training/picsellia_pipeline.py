from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.parameters import (
    AugmentationParameters,
    ExportParameters,
)
from picsellia_cv_engine.core.services.utils.picsellia_context import (
    create_picsellia_training_context,
)
from picsellia_cv_engine.steps.base.dataset.loader import load_yolo_datasets
from picsellia_cv_engine.steps.base.model.builder import build_model
from steps import train
from utils.parameters import TrainingHyperParameters

context = create_picsellia_training_context(
    hyperparameters_cls=TrainingHyperParameters,
    augmentation_parameters_cls=AugmentationParameters,
    export_parameters_cls=ExportParameters,
)


@pipeline(context=context, log_folder_path="logs/", remove_logs_on_completion=False)
def training_pipeline():
    picsellia_datasets = load_yolo_datasets()
    picsellia_model = build_model(pretrained_weights_name="pretrained-weights")
    train(picsellia_model=picsellia_model, picsellia_datasets=picsellia_datasets)


if __name__ == "__main__":
    training_pipeline()

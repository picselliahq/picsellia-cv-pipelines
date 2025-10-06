import argparse

from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.core.services.context.unified_context import (
    create_training_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from steps import (
    evaluate_paddle_ocr_model_collection,
    export_paddle_ocr_model_collection,
    get_paddle_ocr_model_collection,
    load_paddle_ocr_model_collection,
    prepare_paddle_ocr_dataset_collection,
    prepare_paddle_ocr_model_collection,
    train_paddle_ocr_model_collection,
)
from utils.parameters import (
    PaddleOCRAugmentationParameters,
    PaddleOCRHyperParameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_training_context_from_config(
    hyperparameters_cls=PaddleOCRHyperParameters,
    augmentation_parameters_cls=PaddleOCRAugmentationParameters,
    export_parameters_cls=ExportParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def paddle_ocr_training_pipeline():
    dataset_collection = load_coco_datasets()
    dataset_collection = prepare_paddle_ocr_dataset_collection(
        dataset_collection=dataset_collection
    )
    model_collection = get_paddle_ocr_model_collection()
    model_collection = prepare_paddle_ocr_model_collection(
        model_collection=model_collection, dataset_collection=dataset_collection
    )
    model_collection = train_paddle_ocr_model_collection(
        model_collection=model_collection
    )
    model_collection = export_paddle_ocr_model_collection(
        model_collection=model_collection
    )
    model_collection = load_paddle_ocr_model_collection(
        model_collection=model_collection
    )
    evaluate_paddle_ocr_model_collection(
        model_collection=model_collection, dataset=dataset_collection["test"]
    )


if __name__ == "__main__":
    paddle_ocr_training_pipeline()

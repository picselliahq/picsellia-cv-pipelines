# type: ignore
from argparse import ArgumentParser

from picsellia_cv_engine.core.contexts import LocalTrainingContext
from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets

from pipelines.paddle_ocr.pipeline_utils.parameters.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from pipelines.paddle_ocr.pipeline_utils.parameters.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from pipelines.paddle_ocr.pipeline_utils.steps.data_preparation.paddle_ocr_data_preparator import (
    prepare_paddle_ocr_dataset_collection,
)
from pipelines.paddle_ocr.pipeline_utils.steps.model_evaluation.paddle_ocr_model_evaluator import (
    evaluate_paddle_ocr_model_collection,
)
from pipelines.paddle_ocr.pipeline_utils.steps.model_export.paddle_ocr_model_exporter import (
    export_paddle_ocr_model_collection,
)
from pipelines.paddle_ocr.pipeline_utils.steps.model_loading.paddle_ocr_model_collection_loader import (
    load_paddle_ocr_model_collection,
)
from pipelines.paddle_ocr.pipeline_utils.steps.model_training.paddle_ocr_trainer import (
    train_paddle_ocr_model_collection,
)
from pipelines.paddle_ocr.pipeline_utils.steps.weights_extraction.paddle_ocr_weights_extractor import (
    get_paddle_ocr_model_collection,
)
from pipelines.paddle_ocr.pipeline_utils.steps.weights_preparation.paddle_ocr_weights_preparator import (
    prepare_paddle_ocr_model_collection,
)

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
        hyperparameters_cls=PaddleOCRHyperParameters,
        augmentation_parameters_cls=PaddleOCRAugmentationParameters,
        export_parameters_cls=ExportParameters,
    )


@pipeline(
    context=get_context(),
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

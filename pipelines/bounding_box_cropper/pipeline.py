import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset
from steps import process, validate_bounding_box_cropper_data
from utils.parameters import ProcessingBoundingBoxCropperParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.DATASET_VERSION_CREATION,
    processing_parameters_cls=ProcessingBoundingBoxCropperParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def bounding_box_cropper_processing_pipeline() -> None:
    dataset_collection = load_coco_datasets()
    validate_bounding_box_cropper_data(dataset=dataset_collection["input"])
    output_dataset = process(dataset_collection=dataset_collection)
    upload_full_dataset(dataset=output_dataset)


if __name__ == "__main__":
    bounding_box_cropper_processing_pipeline()

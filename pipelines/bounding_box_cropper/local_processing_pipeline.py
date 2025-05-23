# type: ignore
from argparse import ArgumentParser

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset

from pipelines.bounding_box_cropper.pipeline_utils.steps.data_validation.processing_bounding_box_cropper_data_validator import (
    validate_bounding_box_cropper_data,
)
from pipelines.bounding_box_cropper.pipeline_utils.steps.processing.bounding_box_cropper_processing import (
    process,
)

parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_name", type=str)
parser.add_argument("--job_id", type=str)
parser.add_argument("--input_dataset_version_id", type=str)
parser.add_argument("--output_dataset_version_name", type=str, default="output")
parser.add_argument("--label_name_to_extract", type=str, default="person")
parser.add_argument("--datalake", type=str, default="default")
parser.add_argument("--data_tag", type=str, default=None)
parser.add_argument("--fix_annotation", action="store_true", default=False)

args = parser.parse_args()

local_context = create_local_processing_context(
    api_token=args.api_token,
    organization_name=args.organization_name,
    job_type=ProcessingType.DATASET_VERSION_CREATION,
    input_dataset_version_id=args.input_dataset_version_id,
    output_dataset_version_name=args.output_dataset_version_name,
    processing_parameters={
        "label_name_to_extract": args.label_name_to_extract,
        "datalake": args.datalake,
        "data_tag": args.data_tag,
        "fix_annotation": args.fix_annotation,
    },
)


@pipeline(
    context=local_context,
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

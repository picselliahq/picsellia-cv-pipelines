from argparse import ArgumentParser

from pipeline_utils.steps.data_validation.processing_tiler_data_validator import (
    validate_tiler_data,
)
from pipeline_utils.steps.processing.tiler_processing import (
    process,
)
from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_dataset_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset
from pipeline_utils.parameters.processing_tiler_parameters import (
    ProcessingTilerParameters,
)

parser = ArgumentParser()
parser.add_argument("--api_token", required=True, type=str, help="Picsellia API token")
parser.add_argument(
    "--organization_name", required=True, type=str, help="Picsellia organization name"
)
parser.add_argument(
    "--job_type",
    required=True,
    type=str,
    choices=["DATASET_VERSION_CREATION", "PRE_ANNOTATION", "TRAINING"],
    help="Job type",
)
parser.add_argument(
    "--input_dataset_version_id",
    required=True,
    type=str,
    help="Input dataset version ID",
)
parser.add_argument("--output_dataset_version_name", type=str, default="output")
parser.add_argument(
    "--working_dir", required=False, type=str, help="Working directory", default=None
)
args = parser.parse_args()

context = create_local_dataset_processing_context(
    processing_parameters_cls=ProcessingTilerParameters,
    api_token=args.api_token,
    organization_name=args.organization_name,
    job_type=args.job_type,
    input_dataset_version_id=args.input_dataset_version_id,
    output_dataset_version_name=args.output_dataset_version_name,
    working_dir=args.working_dir,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def tiler_processing_pipeline() -> None:
    dataset_collection = load_coco_datasets()
    dataset_collection["input"] = validate_tiler_data(
        dataset=dataset_collection["input"]
    )
    output_dataset = process(dataset_collection=dataset_collection)
    upload_full_dataset(
        dataset=output_dataset,
        use_id=False,
        fail_on_asset_not_found=False,
    )


if __name__ == "__main__":
    tiler_processing_pipeline()

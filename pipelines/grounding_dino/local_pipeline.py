import argparse

from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_dataset_annotations
from picsellia_cv_engine.steps.base.model.builder import build_model

from pipelines.grounding_dino.steps import process
from pipelines.grounding_dino.utils.parameters import GroundingDinoProcessingParameters

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the local processing pipeline")
parser.add_argument("--api_token", required=True, type=str, help="Picsellia API token")
parser.add_argument(
    "--organization_name", required=True, type=str, help="Picsellia Organization ID"
)
parser.add_argument(
    "--job_type",
    required=True,
    type=str,
    choices=["DATASET_VERSION_CREATION", "TRAINING"],
    help="Job type",
)
parser.add_argument(
    "--input_dataset_version_id",
    required=True,
    type=str,
    help="Input dataset version ID",
)
parser.add_argument(
    "--output_dataset_version_name",
    required=False,
    type=str,
    help="Output dataset version name",
    default=None,
)
parser.add_argument(
    "--working_dir", required=False, type=str, help="Working directory", default=None
)
args = parser.parse_args()

# Create local processing context
context = create_local_processing_context(
    processing_parameters_cls=GroundingDinoProcessingParameters,
    api_token=args.api_token,
    organization_name=args.organization_name,
    job_type=args.job_type,
    input_dataset_version_id=args.input_dataset_version_id,
    output_dataset_version_name=args.output_dataset_version_name,
    model_version_id="0196cf7c-11fe-797d-9c55-d1b4a59a594c",
    working_dir=args.working_dir,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def grounding_dino_pipeline():
    picsellia_dataset = load_coco_datasets()
    picsellia_model = build_model(
        pretrained_weights_name="pretrained-weights", config_name="config"
    )
    picsellia_dataset = process(
        picsellia_model=picsellia_model, picsellia_dataset=picsellia_dataset["input"]
    )
    upload_dataset_annotations(dataset=picsellia_dataset, use_id=True)


if __name__ == "__main__":
    grounding_dino_pipeline()

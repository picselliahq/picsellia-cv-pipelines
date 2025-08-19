import argparse

from diversified_dataset_extractor.pipeline_utils.parameters.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from diversified_dataset_extractor.pipeline_utils.steps.data_validation.processing_diversified_data_extractor_data_validator import (
    validate_diversified_data_extractor_data,
)
from diversified_dataset_extractor.pipeline_utils.steps.model_loading.processing_diversified_data_extractor_model_loader import (
    load_diversified_data_extractor_model,
)
from diversified_dataset_extractor.pipeline_utils.steps.processing.diversified_data_extractor_processing import (
    process,
)
from diversified_dataset_extractor.pipeline_utils.steps.weights_validation.processing_diversified_data_extractor_weights_validator import (
    validate_diversified_data_extractor_weights,
)
from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_dataset_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets

parser = argparse.ArgumentParser(description="Run the local preannotation pipeline")
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
    processing_parameters_cls=ProcessingDiversifiedDataExtractorParameters,
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
def diversified_data_extractor_pipeline() -> None:
    datasets = load_coco_datasets(skip_asset_listing=True)

    validate_diversified_data_extractor_data(dataset=datasets["input"])
    pretrained_weights = validate_diversified_data_extractor_weights()
    embedding_model = load_diversified_data_extractor_model(
        pretrained_weights=pretrained_weights
    )

    process(
        input_dataset=datasets["input"],
        output_dataset=datasets["output"],
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    diversified_data_extractor_pipeline()

import argparse

from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_dataset_annotations
from picsellia_cv_engine.steps.base.model.prediction_converter import (
    convert_predictions_to_coco,
)
from picsellia_cv_engine.steps.grounding_dino.model.loader import (
    load_grounding_dino_model,
)
from picsellia_cv_engine.steps.grounding_dino.model.predictor import (
    run_grounding_dino_inference,
)
from utils.parameters import ProcessingParameters

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
parser.add_argument(
    "--model_version_id", required=True, type=str, help="Model version ID"
)
parser.add_argument(
    "--working_dir", required=False, type=str, help="Working directory", default=None
)
args = parser.parse_args()

context = create_local_processing_context(
    processing_parameters_cls=ProcessingParameters,
    api_token=args.api_token,
    organization_name=args.organization_name,
    job_type=args.job_type,
    input_dataset_version_id=args.input_dataset_version_id,
    model_version_id=args.model_version_id,
    working_dir=args.working_dir,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def grounding_dino_pipeline():
    picsellia_dataset = load_coco_datasets()
    grounding_dino_model = load_grounding_dino_model(
        pretrained_weights_name="pretrained-weights", config_name="config"
    )
    predictions = run_grounding_dino_inference(
        model=grounding_dino_model,
        dataset=picsellia_dataset,
    )
    picsellia_dataset = convert_predictions_to_coco(
        predictions=predictions,
        dataset=picsellia_dataset,
    )
    upload_dataset_annotations(dataset=picsellia_dataset, use_id=True)


if __name__ == "__main__":
    grounding_dino_pipeline()

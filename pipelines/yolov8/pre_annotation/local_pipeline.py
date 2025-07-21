from argparse import ArgumentParser

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_processing_context,
)
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_dataset_annotations
from pre_annotation.pipeline_utils.parameters.processing_yolov8_preannotation_parameters import (
    ProcessingYOLOV8PreannotationParameters,
)
from pre_annotation.pipeline_utils.steps.model_loading.processing_ultralytics_model_loader import (
    load_processing_ultralytics_model,
)
from pre_annotation.pipeline_utils.steps.processing.yolov8_preannotation_processing import (
    process,
)
from pre_annotation.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_processing_ultralytics_model,
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
parser.add_argument(
    "--model_version_id", required=True, type=str, help="Model version ID"
)
parser.add_argument("--model_file_name", type=str)
parser.add_argument("--confidence_threshold", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--image_size", type=int, default=640)
parser.add_argument("--label_matching_strategy", type=str, default="add")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--working_dir", required=False, type=str, help="Working directory", default=None
)

args = parser.parse_args()

context = create_local_processing_context(
    processing_parameters_cls=ProcessingYOLOV8PreannotationParameters,
    api_token=args.api_token,
    organization_name=args.organization_name,
    job_type=args.job_type,
    input_dataset_version_id=args.input_dataset_version_id,
    model_version_id=args.model_version_id,
    working_dir=args.working_dir,
)
# local_context.model_version = local_context.client.get_public_model(name="YoloV8-Segmentation").get_version(version="YoloV8-m-segmentation")
context.processing_parameters.agnostic_nms = True
context.processing_parameters.replace_annotations = False


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_preannotation_processing_pipeline() -> None:
    dataset = load_coco_datasets()
    model = get_processing_ultralytics_model()
    load_processing_ultralytics_model(
        model=model,
        weights_path_to_load=model.trained_weights_path,
    )
    output_dataset = process(model=model, dataset=dataset)
    upload_dataset_annotations(dataset=output_dataset)


if __name__ == "__main__":
    yolov8_preannotation_processing_pipeline()

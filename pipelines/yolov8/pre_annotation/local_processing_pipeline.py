from argparse import ArgumentParser

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_processing_context,
)
from picsellia_cv_engine.core.steps.dataset.loader import load_coco_datasets
from picsellia_cv_engine.core.steps.dataset.uploader import upload_dataset_annotations

from pipelines.yolov8.pre_annotation.pipeline_utils.steps.model_loading.processing_ultralytics_model_loader import (
    load_processing_ultralytics_model,
)
from pipelines.yolov8.pre_annotation.pipeline_utils.steps.processing.yolov8_preannotation_processing import (
    process,
)
from pipelines.yolov8.pre_annotation.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_processing_ultralytics_model,
)

parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_name", type=str)
parser.add_argument("--input_dataset_version_id", type=str)
parser.add_argument("--model_version_id", type=str)
parser.add_argument("--model_file_name", type=str)
parser.add_argument("--confidence_threshold", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--image_size", type=int, default=640)
parser.add_argument("--label_matching_strategy", type=str, default="add")
parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()

local_context = create_local_processing_context(
    api_token=args.api_token,
    organization_name=args.organization_name,
    job_type=ProcessingType.PRE_ANNOTATION,
    input_dataset_version_id=args.input_dataset_version_id,
    model_version_id=args.model_version_id,
    processing_parameters={
        "model_file_name": args.model_file_name,
        "confidence_threshold": args.confidence_threshold,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "label_matching_strategy": args.label_matching_strategy,
        "device": "cuda",
        "agnostic_nms": True,
        "replace_annotations": False,
    },
)
# local_context.model_version = local_context.client.get_public_model(name="YoloV8-Segmentation").get_version(version="YoloV8-m-segmentation")
local_context.processing_parameters.agnostic_nms = True
local_context.processing_parameters.replace_annotations = False


@pipeline(
    context=local_context,
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

# type: ignore
from argparse import ArgumentParser

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.utils.local_context import (
    create_local_processing_context,
)
from picsellia_cv_engine.steps.data_extraction.processing_data_extractor import (
    get_processing_dataset_context,
)
from picsellia_cv_engine.steps.data_upload.annotations_uploader import (
    upload_annotations,
)

from pipelines.yolov8_preannotation.pipeline_utils.steps.model_loading.processing_ultralytics_model_loader import (
    load_processing_ultralytics_model_context,
)
from pipelines.yolov8_preannotation.pipeline_utils.steps.processing.yolov8_preannotation_processing import (
    process,
)
from pipelines.yolov8_preannotation.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_processing_ultralytics_model_context,
)

parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--job_id", type=str)
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
    organization_id=args.organization_id,
    job_id=args.job_id,
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
    },
)


@pipeline(
    context=local_context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_preannotation_processing_pipeline() -> None:
    dataset_context = get_processing_dataset_context()
    model_context = get_processing_ultralytics_model_context()
    load_processing_ultralytics_model_context(
        model_context=model_context,
        weights_path_to_load=model_context.trained_weights_path,
    )
    output_dataset_context = process(
        model_context=model_context, dataset_context=dataset_context
    )
    upload_annotations(dataset_context=output_dataset_context)


if __name__ == "__main__":
    yolov8_preannotation_processing_pipeline()

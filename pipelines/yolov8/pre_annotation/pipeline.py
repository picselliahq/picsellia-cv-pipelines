import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_dataset_annotations
from steps import get_model, load_model, process
from utils.parameters import (
    ProcessingYOLOV8PreannotationParameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.PRE_ANNOTATION,
    processing_parameters_cls=ProcessingYOLOV8PreannotationParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_preannotation_processing_pipeline() -> None:
    dataset = load_coco_datasets()
    model = get_model()
    load_model(
        model=model,
        weights_path_to_load=model.trained_weights_path,
    )
    output_dataset = process(model=model, dataset=dataset)
    upload_dataset_annotations(dataset=output_dataset)


if __name__ == "__main__":
    yolov8_preannotation_processing_pipeline()

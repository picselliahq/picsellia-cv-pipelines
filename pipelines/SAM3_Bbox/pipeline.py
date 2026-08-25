import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from steps import load_sam3_model, process, upload_annotations
from utils.parameters import ProcessingParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.PRE_ANNOTATION,
    processing_parameters_cls=ProcessingParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def sam3_labeling_pipeline():
    """
    SAM-3 Labeling Pipeline

    This pipeline:
    1. Loads the dataset from Picsellia
    2. Loads the SAM-3 model from Hugging Face
    3. Processes images with SAM-3 segmentation (supports multi-class)
    4. Uploads annotations back to Picsellia

    Required parameters in config:
    - text_prompt: Text description. Supports multi-class via comma-separated values.
                   Examples:
                   - Single class: "waste"
                   - Multi-class: "car,person,bicycle"
    - threshold: Detection confidence threshold (default: 0.3)
    - mask_threshold: Mask confidence threshold (default: 0.5)

    Optional parameters:
    - min_area: Minimum mask area in pixels (default: 50.0)
    - max_overlap_ratio: Maximum overlap ratio for same-class deduplication (default: 0.3)

    Multi-class deduplication parameters:
    - iou_threshold: IoU threshold for cross-class overlap detection (default: 0.5)
    - containment_threshold: Threshold for nested mask detection (default: 0.8)
    - deduplication_strategy: "keep_smaller" or "keep_larger" (default: "keep_smaller")
                             keep_smaller: Prioritize smaller, more precise masks
                             keep_larger: Prioritize larger, more complete masks
    - annotation_mode: "keep", "replace" or "concatenate" (default: "concatenate")
                        How to handle annotations already present on an asset.
    """
    # Load dataset
    picsellia_dataset = load_coco_datasets()

    # Load SAM-3 model from Hugging Face (no need for build_model)
    model, processor = load_sam3_model()

    # Process images with SAM-3
    picsellia_dataset = process(
        model=model, processor=processor, picsellia_dataset=picsellia_dataset
    )

    # Upload annotations to Picsellia
    upload_annotations(picsellia_dataset=picsellia_dataset)


if __name__ == "__main__":
    sam3_labeling_pipeline()

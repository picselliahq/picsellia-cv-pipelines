import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from steps import fork_dataset, load_sam3_model, process, upload_annotations
from utils.parameters import ProcessingParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.DATASET_VERSION_CREATION,
    processing_parameters_cls=ProcessingParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def sam3_polygons_from_box_pipeline():
    """
    SAM-3 Polygons-from-Box Pipeline

    This pipeline:
    1. Forks the input (bbox-annotated) dataset into a new SEGMENTATION
       dataset version, without re-uploading images
    2. Loads the SAM-3 model from Hugging Face
    3. For every existing bounding box annotation, prompts SAM-3 with that
       exact box to generate a precise polygon mask - the resulting polygon
       keeps the SAME label as the box it was generated from
    4. Uploads the polygon annotations onto the new dataset version

    Required inputs:
    - output_dataset_version_name: Name of the new dataset version to create

    Optional parameters:
    - threshold: Detection confidence threshold applied to SAM-3's output (default: 0.3)
    - mask_threshold: Mask confidence threshold, controls how tight/loose
                       polygons are (default: 0.5)
    - min_area: Minimum polygon area in pixels; masks smaller than this are
                discarded (default: 10.0)
    - fallback_to_bbox_polygon: If SAM-3 fails to produce a mask for a box,
                                 fall back to a rectangular polygon matching
                                 that box so every input box still gets a
                                 corresponding output polygon (default: True)
    - annotation_mode: "keep", "replace" or "concatenate" (default: "replace")
                        How to handle annotations already present on the
                        output dataset version.
    """
    output_dataset_version = fork_dataset()

    model, processor = load_sam3_model()

    coco_file_path = process(
        model=model,
        processor=processor,
        output_dataset_version=output_dataset_version,
    )

    upload_annotations(
        coco_file_path=coco_file_path, output_dataset_version=output_dataset_version
    )


if __name__ == "__main__":
    sam3_polygons_from_box_pipeline()

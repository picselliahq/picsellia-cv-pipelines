from argparse import ArgumentParser

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.utils.local_context import (
    create_local_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset

from pipelines.dataset_tiler.pipeline_utils.steps.data_validation.processing_tiler_data_validator import (
    validate_tiler_data,
)
from pipelines.dataset_tiler.pipeline_utils.steps.processing.tiler_processing import (
    process,
)
from pipelines.dataset_tiler.pipeline_utils.steps_utils.processing.base_tiler_processing import (
    TileMode,
)

parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_name", type=str)
parser.add_argument("--job_id", type=str)
parser.add_argument("--input_dataset_version_id", type=str)
parser.add_argument("--output_dataset_version_name", type=str)
parser.add_argument("--tile_height", type=int, default=640)
parser.add_argument("--tile_width", type=int, default=640)
parser.add_argument("--overlap_height_ratio", type=float, default=0.1)
parser.add_argument("--overlap_width_ratio", type=float, default=0.1)
parser.add_argument("--min_annotation_area_ratio", type=float, default=0.1)
parser.add_argument("--min_annotation_width", type=int, default=0)
parser.add_argument("--min_annotation_height", type=int, default=0)
parser.add_argument("--padding_color_value", type=int, default=114)
parser.add_argument("--datalake", type=str, default="default")
parser.add_argument("--data_tag", type=str)
parser.add_argument("--fix_annotation", action="store_true", default=False)

args = parser.parse_args()

local_context = create_local_processing_context(
    api_token=args.api_token,
    organization_name=args.organization_name,
    job_type=ProcessingType.DATASET_VERSION_CREATION,
    input_dataset_version_id=args.input_dataset_version_id,
    output_dataset_version_name=args.output_dataset_version_name,
    processing_parameters={
        "tiling_mode": "constant",
        "tile_height": args.tile_height,
        "tile_width": args.tile_width,
        "overlap_height_ratio": args.overlap_height_ratio,
        "overlap_width_ratio": args.overlap_width_ratio,
        "min_annotation_area_ratio": args.min_annotation_area_ratio,
        "min_annotation_width": args.min_annotation_width,
        "min_annotation_height": args.min_annotation_height,
        "padding_color_value": args.padding_color_value,
        "datalake": args.datalake,
        "data_tag": args.data_tag,
        "fix_annotation": args.fix_annotation,
    },
)
local_context.processing_parameters.tiling_mode = TileMode.CONSTANT


@pipeline(
    context=local_context,
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

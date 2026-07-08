import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import create_processing_context_from_config
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline

from steps import (
    download_videos,
    split_videos,
    upload_annotations,
    upload_segments_and_create_dataset,
)
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
def split_video_pipeline():
    video_assets, videos_dir, video_coco_data, video_tags = download_videos()
    segment_metadata = split_videos(video_assets, videos_dir)
    new_dataset_version = upload_segments_and_create_dataset(segment_metadata, video_tags)
    upload_annotations(new_dataset_version, video_coco_data, segment_metadata)


if __name__ == "__main__":
    split_video_pipeline()

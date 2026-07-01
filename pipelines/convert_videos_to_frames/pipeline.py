import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import create_processing_context_from_config
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline

from steps import download_videos, extract_frames, upload_annotations, upload_frames_and_create_dataset
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
def convert_videos_to_frames_pipeline():
    video_assets, videos_dir, video_coco_data = download_videos()
    frames_dir, frames_coco, frame_to_video = extract_frames(
        video_assets, videos_dir, video_coco_data
    )
    new_dataset_version = upload_frames_and_create_dataset(
        frames_dir, frames_coco, frame_to_video
    )
    upload_annotations(new_dataset_version, frames_coco)


if __name__ == "__main__":
    convert_videos_to_frames_pipeline()

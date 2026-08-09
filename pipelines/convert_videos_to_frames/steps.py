import json
import os
from datetime import datetime
from typing import Any

from picsellia import DatasetVersion
from picsellia.exceptions import ResourceConflictError
from picsellia.types.enums import AnnotationFileType, DataType, ImportAnnotationMode
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.processing import detect_inference_type, extract_frames_and_build_coco


@step
def download_videos() -> tuple[list, str, dict[str, Any]]:
    """
    List video assets in the input dataset version, download them locally,
    and export their COCO annotations in video format.

    Returns:
        video_assets: list of video Asset objects.
        videos_dir: local directory containing the downloaded video files.
        video_coco_data: parsed video COCO dict (with 'videos', 'images', 'annotations').
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    dataset_version = context.target

    if context.asset_ids:
        assets = dataset_version.list_assets(ids=list(context.asset_ids))
    else:
        assets = dataset_version.list_assets()

    video_assets = [asset for asset in assets if asset.type == DataType.VIDEO]
    if not video_assets:
        raise ValueError(
            "No video assets found in the selected dataset version. "
            "Only VIDEO-type assets are supported by this pipeline."
        )

    videos_dir = os.path.join(context.working_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    for asset in video_assets:
        asset.download(target_path=videos_dir, force_replace=True)
    print(f"Downloaded {len(video_assets)} video(s) to '{videos_dir}'.")

    coco_dir = os.path.join(context.working_dir, "annotations", "input")
    os.makedirs(coco_dir, exist_ok=True)
    coco_path = dataset_version.export_annotation_file(
        annotation_file_type=AnnotationFileType.COCO,
        target_path=coco_dir,
        assets=video_assets,
        export_video=True,
        use_id=False,
    )
    with open(coco_path) as f:
        video_coco_data = json.load(f)

    print(
        f"Exported video COCO annotations: "
        f"{len(video_coco_data.get('videos', []))} video(s), "
        f"{len(video_coco_data.get('images', []))} annotated frame(s), "
        f"{len(video_coco_data.get('annotations', []))} annotation(s)."
    )
    return video_assets, videos_dir, video_coco_data


@step
def extract_frames(
    video_assets: list,
    videos_dir: str,
    video_coco_data: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """
    Extract every frame from each downloaded video using OpenCV and build a
    standard image COCO file (track annotations converted to per-frame ones).

    Returns:
        frames_dir: directory containing the extracted JPEG frames.
        frames_coco: standard COCO dict (images = all frames, converted annotations).
        frame_to_video: {frame_filename: origin_video_filename}.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    frames_dir = os.path.join(context.working_dir, "frames")

    frames_coco, frame_to_video = extract_frames_and_build_coco(
        video_coco_data=video_coco_data,
        videos_dir=videos_dir,
        frames_dir=frames_dir,
    )
    return frames_dir, frames_coco, frame_to_video


@step
def upload_frames_and_create_dataset(
    frames_dir: str,
    frames_coco: dict[str, Any],
    frame_to_video: dict[str, str],
) -> DatasetVersion:
    """
    Upload all extracted frames to the datalake (with tag 'video_frames' and
    the origin video filename as custom metadata), create a new dataset version
    on the same parent dataset, and add the uploaded frames to it.

    Returns the newly created DatasetVersion.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters

    version_name = context.inputs.get("output_dataset_version_name")
    if not version_name:
        raise ValueError("Input 'output_dataset_version_name' is required.")

    datalake = context.client.get_datalake(id=context.inputs.get("datalake"))

    frame_images = frames_coco.get("images", [])

    frame_filepaths = [
        os.path.join(frames_dir, img["file_name"]) for img in frame_images
    ]
    frame_custom_metadata = [
        {"origin_video": frame_to_video[img["file_name"]]} for img in frame_images
    ]

    print(f"Uploading {len(frame_filepaths)} frame(s) to the datalake…")
    uploaded_data = datalake.upload_data(
        filepaths=frame_filepaths,
        tags=["video_frames"],
        custom_metadata=frame_custom_metadata,
    )

    parent_dataset = context.client.get_dataset(name=context.target.name)
    try:
        new_dataset_version = parent_dataset.create_version(version=version_name)
    except ResourceConflictError:
        version_name = f"{version_name}_{datetime.now().timestamp()}"
        print(
            f"A dataset version with that name already exists, "
            f"creating as '{version_name}' instead."
        )
        new_dataset_version = parent_dataset.create_version(version=version_name)

    job = new_dataset_version.add_data(data=uploaded_data, wait=False)
    job.wait_for_done(attempts=1000)

    print(
        f"Created dataset version '{new_dataset_version.version}' with "
        f"{len(frame_filepaths)} frame(s)."
    )
    return new_dataset_version


@step
def upload_annotations(
    new_dataset_version: DatasetVersion,
    frames_coco: dict[str, Any],
) -> None:
    """
    Detect the inference type from the converted COCO, configure the new
    dataset version, and import the frame annotations.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()

    annotations = frames_coco.get("annotations", [])
    if not annotations:
        print("No annotations to import — dataset version created without annotations.")
        return

    inference_type = detect_inference_type(frames_coco)
    new_dataset_version.set_type(inference_type)

    output_dir = os.path.join(context.working_dir, "annotations", "output")
    os.makedirs(output_dir, exist_ok=True)
    coco_path = os.path.join(output_dir, "frames_coco.json")
    with open(coco_path, "w") as f:
        json.dump(frames_coco, f)

    new_dataset_version.import_annotations_coco_file(
        file_path=coco_path,
        use_id=False,
        mode=ImportAnnotationMode.REPLACE,
        fail_on_asset_not_found=False,
    )
    print(
        f"Imported {len(annotations)} annotation(s) into "
        f"'{new_dataset_version.version}'."
    )

import json
import os
from datetime import datetime
from typing import Any

from picsellia import DatasetVersion
from picsellia.exceptions import ResourceConflictError
from picsellia.types.enums import (
    AnnotationFileType,
    DataType,
    ImportAnnotationMode,
    InferenceType,
)
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.processing import (
    build_segment_video_coco,
    detect_inference_type,
    split_videos_into_segments,
)

UPLOAD_BATCH_SIZE = 100


@step
def download_videos() -> tuple[list, str, dict[str, Any], dict[str, list[str]]]:
    """
    List video assets in the input dataset version, download them locally,
    export their COCO annotations in video format (if any), and read each
    video's tags.

    Returns:
        video_assets: list of video Asset objects.
        videos_dir: local directory containing the downloaded video files.
        video_coco_data: parsed video COCO dict (with 'videos', 'images', 'annotations').
        video_tags: {video_filename: [tag_name, ...]}.
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

    video_tags: dict[str, list[str]] = {}
    for asset in video_assets:
        asset.download(target_path=videos_dir, force_replace=True)
        video_tags[asset.filename] = [tag.name for tag in asset.get_tags()]
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
        f"{len(video_coco_data.get('annotations', []))} annotation(s)."
    )
    return video_assets, videos_dir, video_coco_data, video_tags


@step
def split_videos(video_assets: list, videos_dir: str) -> dict[str, dict[str, Any]]:
    """
    Cut every downloaded video into consecutive segments of
    `frames_per_segment` (input) frames each.

    Returns:
        segment_metadata: {segment_filepath: {"source_video": origin_video_filename,
        "start_frame": int, "end_frame": int}}.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()

    frames_per_segment = int(context.inputs.get("frames_per_segment"))
    if frames_per_segment <= 0:
        raise ValueError("Input 'frames_per_segment' must be > 0.")

    segments_dir = os.path.join(context.working_dir, "segments")
    segment_metadata = split_videos_into_segments(
        video_filenames=[asset.filename for asset in video_assets],
        videos_dir=videos_dir,
        output_dir=segments_dir,
        frames_per_segment=frames_per_segment,
    )

    if not segment_metadata:
        raise ValueError("No video segments were produced.")

    return segment_metadata


@step
def upload_segments_and_create_dataset(
    segment_metadata: dict[str, dict[str, Any]],
    video_tags: dict[str, list[str]],
) -> DatasetVersion:
    """
    Upload all video segments to the datalake (with tag 'video_segment' and,
    as custom metadata, the origin video filename as 'source_video' and its
    tags as 'source_video_tags'), create a new dataset version on the same
    parent dataset, and add the uploaded segments to it.

    Returns the newly created DatasetVersion.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()

    version_name = context.inputs.get("output_dataset_version_name")
    if not version_name:
        raise ValueError("Input 'output_dataset_version_name' is required.")

    datalake = context.client.get_datalake(name=context.inputs.get("datalake"))

    segment_paths = list(segment_metadata.keys())
    segment_custom_metadata = [
        {
            "source_video": segment_metadata[path]["source_video"],
            "source_video_tags": video_tags.get(
                segment_metadata[path]["source_video"], []
            ),
        }
        for path in segment_paths
    ]

    print(f"Uploading {len(segment_paths)} video segment(s) to the datalake…")
    uploaded_data = None
    for batch_start in range(0, len(segment_paths), UPLOAD_BATCH_SIZE):
        batch_paths = segment_paths[batch_start : batch_start + UPLOAD_BATCH_SIZE]
        batch_metadata = segment_custom_metadata[
            batch_start : batch_start + UPLOAD_BATCH_SIZE
        ]
        print(
            f"Uploading batch {batch_start // UPLOAD_BATCH_SIZE + 1} "
            f"({len(batch_paths)} segment(s))…"
        )
        batch_uploaded_data = datalake.upload_data(
            filepaths=batch_paths,
            tags=["video_segment"],
            custom_metadata=batch_metadata,
            wait_for_unprocessable_data=False,
        )
        batch_uploaded_data.wait_for_upload_done(
            blocking_time_increment=60.0, attempts=60
        )
        if uploaded_data is None:
            uploaded_data = batch_uploaded_data
        else:
            uploaded_data += batch_uploaded_data

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
        f"{len(segment_paths)} video segment(s)."
    )
    return new_dataset_version


@step
def upload_annotations(
    new_dataset_version: DatasetVersion,
    video_coco_data: dict[str, Any],
    segment_metadata: dict[str, dict[str, Any]],
) -> None:
    """
    Remap the original per-video track annotations onto the new segments and
    import them as video-track annotations on the new dataset version.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()

    if not video_coco_data.get("annotations"):
        print(
            "No annotations on the source video(s) — dataset version created without annotations."
        )
        return

    segment_video_coco = build_segment_video_coco(
        video_coco_data=video_coco_data,
        segment_metadata=segment_metadata,
    )
    annotations = segment_video_coco.get("annotations", [])
    if not annotations:
        print(
            "No annotation could be remapped onto the produced segments — skipping import."
        )
        return

    inference_type = detect_inference_type(segment_video_coco)
    if inference_type not in (
        InferenceType.OBJECT_DETECTION,
        InferenceType.SEGMENTATION,
    ):
        print(
            f"Annotation type '{inference_type}' is not supported for video-track "
            f"import — skipping annotation import."
        )
        return

    new_dataset_version.set_type(inference_type)

    output_dir = os.path.join(context.working_dir, "annotations", "output")
    os.makedirs(output_dir, exist_ok=True)
    coco_path = os.path.join(output_dir, "segments_coco.json")
    with open(coco_path, "w") as f:
        json.dump(segment_video_coco, f)

    new_dataset_version.import_annotations_coco_video_file(
        file_path=coco_path,
        use_id=False,
        mode=ImportAnnotationMode.REPLACE,
    )
    print(
        f"Imported {len(annotations)} annotation(s) into "
        f"'{new_dataset_version.version}'."
    )

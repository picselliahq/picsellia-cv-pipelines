import json
import logging
import os
from collections import defaultdict

from picsellia.types.enums import AnnotationFileType
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from utils.parameters import ProcessingParameters
from utils.processing import extract_frames_from_all_videos

logger = logging.getLogger("picsellia-engine")


@step
def download_video_annotations(dataset: CocoDataset) -> CocoDataset:
    """
    Download COCO annotations with export_video=True for video assets.
    Populates dataset.coco_data with video-aware annotations.
    """
    annotations_dir = os.path.join(
        os.path.dirname(dataset.images_dir), "video_annotations"
    )
    os.makedirs(annotations_dir, exist_ok=True)

    coco_path = dataset.dataset_version.export_annotation_file(
        annotation_file_type=AnnotationFileType.COCO,
        target_path=os.path.join(annotations_dir, "coco_video.json"),
        assets=dataset.assets,
        export_video=True,
        use_id=True,
    )

    with open(coco_path) as f:
        dataset.coco_data = json.load(f)

    logger.info(
        f"Downloaded video annotations: "
        f"{len(dataset.coco_data.get('annotations', []))} annotations, "
        f"{len(dataset.coco_data.get('categories', []))} categories"
    )
    return dataset


def _build_frame_annotations_index(
    coco_data: dict,
) -> tuple[dict[tuple[str, int], list[dict]], list[dict]]:
    """
    Build a mapping from (video_filename, frame_id) to annotations.

    The video COCO export contains:
    - 'videos': list of {"id": int, "file_name": "uuid.mp4"}
    - 'images': list of {"id": int, "video_id": int, "frame_id": int, ...}
    - 'annotations': linked to images via image_id

    Returns:
        A tuple of ((video_filename, frame_id) -> list of annotations, categories list).
    """
    video_id_to_filename = {}
    for video in coco_data.get("videos", []):
        video_id_to_filename[video["id"]] = video["file_name"]

    image_id_to_key: dict[int, tuple[str, int]] = {}
    for image in coco_data.get("images", []):
        video_filename = video_id_to_filename.get(image.get("video_id"))
        if video_filename is not None:
            image_id_to_key[image["id"]] = (video_filename, image["frame_id"])

    frame_annotations: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        key = image_id_to_key.get(ann["image_id"])
        if key:
            frame_annotations[key].append(ann)

    return frame_annotations, coco_data.get("categories", [])


@step
def extract_frames(
    input_dataset: CocoDataset, output_dataset: CocoDataset
) -> CocoDataset:
    """
    Extract frames from video assets in the input dataset and populate the output dataset.
    Annotations from the input videos are propagated to each extracted frame.
    """
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = (
        Pipeline.get_active_context()
    )
    parameters = context.processing_parameters

    frames_metadata = extract_frames_from_all_videos(
        input_dir=input_dataset.images_dir,
        output_dir=output_dataset.images_dir,
        frame_interval=parameters.frame_interval,
        max_frames_per_video=parameters.max_frames_per_video,
    )

    frame_annotations, categories = _build_frame_annotations_index(
        input_dataset.coco_data or {}
    )

    output_coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    annotation_id = 0
    for idx, frame in enumerate(frames_metadata):
        image_id = idx
        output_coco["images"].append(
            {
                "id": image_id,
                "file_name": frame["file_name"],
                "width": frame["width"],
                "height": frame["height"],
            }
        )

        key = (frame.get("source_video", ""), frame.get("frame_index", -1))
        for ann in frame_annotations.get(key, []):
            new_ann = {k: v for k, v in ann.items() if k not in ("id", "image_id")}
            new_ann["id"] = annotation_id
            new_ann["image_id"] = image_id
            output_coco["annotations"].append(new_ann)
            annotation_id += 1

    output_dataset.coco_data = output_coco

    ann_count = len(output_coco["annotations"])
    logger.info(
        f"Frame extraction complete: {len(frames_metadata)} frames, "
        f"{ann_count} annotations propagated"
    )
    return output_dataset

import os
from pathlib import Path
from typing import Any

import cv2
from picsellia.types.enums import InferenceType


def _scale_segmentation(
    segmentation: Any, scale_x: float, scale_y: float
) -> Any:
    """Rescale COCO polygon segmentation coordinates. RLE (dict) segmentations
    are returned unchanged since they encode a mask at a fixed resolution."""
    if scale_x == 1.0 and scale_y == 1.0:
        return segmentation
    if isinstance(segmentation, dict):
        return segmentation
    return [
        [
            coord * (scale_x if i % 2 == 0 else scale_y)
            for i, coord in enumerate(polygon)
        ]
        for polygon in segmentation
    ]


def extract_frames_and_build_coco(
    video_coco_data: dict[str, Any],
    videos_dir: str,
    frames_dir: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Extract every frame from each video referenced in `video_coco_data`.
    Build a standard COCO file where every frame is an image entry, and
    video track annotations are converted to per-frame image annotations.

    Returns:
        frames_coco: standard COCO dict ready for import.
        frame_to_video: {frame_filename: origin_video_filename}.
    """
    os.makedirs(frames_dir, exist_ok=True)

    video_id_to_filename: dict[int, str] = {
        v["id"]: v["file_name"] for v in video_coco_data.get("videos", [])
    }

    # (video_id, frame_id) -> old image_id in the video COCO
    annotated_key_to_old_image_id: dict[tuple[int, int], int] = {}
    # old image_id -> (width, height) as recorded at annotation time
    old_image_dims: dict[int, tuple[float, float]] = {}
    for img in video_coco_data.get("images", []):
        vid = img.get("video_id")
        fid = img.get("frame_id")
        if vid is not None and fid is not None:
            annotated_key_to_old_image_id[(vid, fid)] = img["id"]
        old_image_dims[img["id"]] = (img.get("width"), img.get("height"))

    frames_coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": video_coco_data.get("categories", []),
    }
    frame_to_video: dict[str, str] = {}
    old_image_id_to_new: dict[int, int] = {}
    new_image_dims: dict[int, tuple[int, int]] = {}
    new_image_id = 0

    for vid_id, video_filename in video_id_to_filename.items():
        video_path = os.path.join(videos_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"Video file not found, skipping: {video_path}")
            continue

        video_stem = Path(video_filename).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_filename), frame)

            h, w = frame.shape[:2]
            frames_coco["images"].append(
                {
                    "id": new_image_id,
                    "file_name": frame_filename,
                    "width": w,
                    "height": h,
                }
            )
            frame_to_video[frame_filename] = video_filename
            new_image_dims[new_image_id] = (w, h)

            key = (vid_id, frame_idx)
            if key in annotated_key_to_old_image_id:
                old_image_id_to_new[annotated_key_to_old_image_id[key]] = new_image_id

            new_image_id += 1
            frame_idx += 1

        cap.release()
        print(f"Extracted {frame_idx} frames from '{video_filename}'.")

    # Convert video track annotations to standard per-frame COCO annotations,
    # rescaling coordinates if the actual decoded frame size differs from the
    # image size that was used at annotation time (e.g. video rotation metadata
    # applied by the annotation platform but ignored by OpenCV's decoder).
    new_ann_id = 0
    for ann in video_coco_data.get("annotations", []):
        old_img_id = ann.get("image_id")
        if old_img_id not in old_image_id_to_new:
            continue

        new_img_id = old_image_id_to_new[old_img_id]
        old_w, old_h = old_image_dims.get(old_img_id, (None, None))
        new_w, new_h = new_image_dims[new_img_id]
        scale_x = new_w / old_w if old_w else 1.0
        scale_y = new_h / old_h if old_h else 1.0

        bbox = ann.get("bbox", [])
        if bbox and (scale_x != 1.0 or scale_y != 1.0):
            x, y, bw, bh = bbox
            bbox = [x * scale_x, y * scale_y, bw * scale_x, bh * scale_y]

        area = ann.get("area", 0.0) * scale_x * scale_y

        new_ann: dict[str, Any] = {
            "id": new_ann_id,
            "image_id": new_img_id,
            "category_id": ann["category_id"],
            "bbox": bbox,
            "area": area,
            "iscrowd": ann.get("iscrowd", 0),
        }
        seg = ann.get("segmentation")
        if seg:
            new_ann["segmentation"] = _scale_segmentation(seg, scale_x, scale_y)

        frames_coco["annotations"].append(new_ann)
        new_ann_id += 1

    print(
        f"Built frames COCO: {len(frames_coco['images'])} frames, "
        f"{len(frames_coco['annotations'])} annotations."
    )
    return frames_coco, frame_to_video


def detect_inference_type(coco_data: dict[str, Any]) -> InferenceType:
    """Infer InferenceType from the first annotation in a COCO file."""
    annotations = coco_data.get("annotations", [])
    if not annotations:
        return InferenceType.OBJECT_DETECTION
    first = annotations[0]
    seg = first.get("segmentation")
    if seg and isinstance(seg, dict):
        return InferenceType.MASK
    if seg and isinstance(seg, list) and seg:
        return InferenceType.SEGMENTATION
    if first.get("keypoints"):
        return InferenceType.KEYPOINT
    if first.get("bbox"):
        return InferenceType.OBJECT_DETECTION
    if "category_id" in first:
        return InferenceType.CLASSIFICATION
    return InferenceType.OBJECT_DETECTION

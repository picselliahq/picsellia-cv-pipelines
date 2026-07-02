import os
from pathlib import Path
from typing import Any

import cv2
from picsellia.types.enums import InferenceType


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
    for img in video_coco_data.get("images", []):
        vid = img.get("video_id")
        fid = img.get("frame_id")
        if vid is not None and fid is not None:
            annotated_key_to_old_image_id[(vid, fid)] = img["id"]

    frames_coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": video_coco_data.get("categories", []),
    }
    frame_to_video: dict[str, str] = {}
    old_image_id_to_new: dict[int, int] = {}
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

            key = (vid_id, frame_idx)
            if key in annotated_key_to_old_image_id:
                old_image_id_to_new[annotated_key_to_old_image_id[key]] = new_image_id

            new_image_id += 1
            frame_idx += 1

        cap.release()
        print(f"Extracted {frame_idx} frames from '{video_filename}'.")

    # Convert video track annotations to standard per-frame COCO annotations
    new_ann_id = 0
    for ann in video_coco_data.get("annotations", []):
        old_img_id = ann.get("image_id")
        if old_img_id not in old_image_id_to_new:
            continue

        new_ann: dict[str, Any] = {
            "id": new_ann_id,
            "image_id": old_image_id_to_new[old_img_id],
            "category_id": ann["category_id"],
            "bbox": ann.get("bbox", []),
            "area": ann.get("area", 0.0),
            "iscrowd": ann.get("iscrowd", 0),
        }
        seg = ann.get("segmentation")
        if seg:
            new_ann["segmentation"] = seg

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

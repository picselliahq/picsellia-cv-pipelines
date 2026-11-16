import os
from pathlib import Path
from typing import Any

import cv2
from picsellia.types.enums import InferenceType


def _collect_keyframe_ids(video_coco_data: dict[str, Any]) -> dict[int, set[int]]:
    """
    {video_id: {frame_id, ...}} for frames marked as a track keyframe in
    Picsellia's video annotation export (frame_attributes.keyframe == True) -
    i.e. frames that were manually annotated, as opposed to interpolated
    ones. Several tracks can each have a keyframe on the same frame_id (one
    annotation per track); that frame only needs to be extracted once, so
    results are deduplicated per video via a set.
    """
    image_id_to_video_id = {
        img["id"]: img.get("video_id") for img in video_coco_data.get("images", [])
    }
    keyframe_ids: dict[int, set[int]] = {}
    for ann in video_coco_data.get("annotations", []):
        frame_attributes = ann.get("frame_attributes") or {}
        if not frame_attributes.get("keyframe"):
            continue
        video_id = ann.get("video_id")
        if video_id is None:
            video_id = image_id_to_video_id.get(ann.get("image_id"))
        frame_id = frame_attributes.get("frame_id")
        if video_id is None or frame_id is None:
            continue
        keyframe_ids.setdefault(video_id, set()).add(frame_id)
    return keyframe_ids


def extract_frames_and_build_coco(
    video_coco_data: dict[str, Any],
    videos_dir: str,
    frames_dir: str,
    extract_only_keyframes: bool = False,
    extract_every_n_frames: int = 0,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Extract frames from each video referenced in `video_coco_data`, according
    to `extract_only_keyframes` / `extract_every_n_frames` (see
    `ProcessingParameters` for their semantics - at most one may be active).
    Build a standard COCO file where every extracted frame is an image entry,
    and video track annotations are converted to per-frame image annotations.

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

    keyframe_ids_by_video = (
        _collect_keyframe_ids(video_coco_data) if extract_only_keyframes else {}
    )

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

        frames_to_keep: set[int] | None = None
        if extract_only_keyframes:
            frames_to_keep = keyframe_ids_by_video.get(vid_id, set())
            if not frames_to_keep:
                print(
                    f"No keyframe found for video_id={vid_id} "
                    f"('{video_filename}') — no frame will be extracted."
                )

        video_stem = Path(video_filename).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        extracted_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frames_to_keep is not None:
                keep_frame = frame_idx in frames_to_keep
            elif extract_every_n_frames > 0:
                keep_frame = frame_idx % extract_every_n_frames == 0
            else:
                keep_frame = True

            if not keep_frame:
                frame_idx += 1
                continue

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
            extracted_count += 1
            frame_idx += 1

        cap.release()
        print(f"Extracted {extracted_count} frame(s) from '{video_filename}'.")

    # Convert video track annotations to standard per-frame COCO annotations.
    # Annotations whose frame wasn't extracted (filtered out above) are
    # naturally dropped since their old image_id has no entry in
    # `old_image_id_to_new`.
    new_ann_id = 0
    for ann in video_coco_data.get("annotations", []):
        old_img_id = ann.get("image_id")
        if old_img_id not in old_image_id_to_new:
            continue

        new_img_id = old_image_id_to_new[old_img_id]

        new_ann: dict[str, Any] = {
            "id": new_ann_id,
            "image_id": new_img_id,
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

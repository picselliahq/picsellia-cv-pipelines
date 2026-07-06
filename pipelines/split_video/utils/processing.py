import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from picsellia.types.enums import InferenceType

DEFAULT_FPS = 25.0


@dataclass
class VideoSegment:
    path: str
    start_frame: int  # inclusive, absolute frame index in the source video
    end_frame: int  # exclusive


def split_video_into_segments(
    video_path: str,
    output_dir: str,
    frames_per_segment: int,
) -> list[VideoSegment]:
    """
    Cut a single video file into consecutive segments of `frames_per_segment` frames
    each (the last segment may be shorter, it keeps whatever frames remain).

    Returns the list of output segments, in order, with their absolute
    [start_frame, end_frame) range in the source video.
    """
    video_stem = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print(f"Could not read FPS for '{video_path}', defaulting to {DEFAULT_FPS}.")
        fps = DEFAULT_FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_segment = max(1, frames_per_segment)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    segments: list[VideoSegment] = []
    writer: cv2.VideoWriter | None = None
    segment_path = ""
    segment_start_frame = 0
    frame_idx = 0
    segment_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if writer is None:
            segment_filename = f"{video_stem}_part_{segment_idx:03d}.mp4"
            segment_path = os.path.join(output_dir, segment_filename)
            writer = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
            segment_start_frame = frame_idx

        writer.write(frame)
        frame_idx += 1

        if frame_idx - segment_start_frame >= frames_per_segment:
            writer.release()
            segments.append(
                VideoSegment(
                    path=segment_path, start_frame=segment_start_frame, end_frame=frame_idx
                )
            )
            writer = None
            segment_idx += 1

    if writer is not None:
        writer.release()
        segments.append(
            VideoSegment(
                path=segment_path, start_frame=segment_start_frame, end_frame=frame_idx
            )
        )
    cap.release()

    print(
        f"Split '{os.path.basename(video_path)}' into {len(segments)} "
        f"segment(s) of up to {frames_per_segment} frame(s) each."
    )
    return segments


def split_videos_into_segments(
    video_filenames: list[str],
    videos_dir: str,
    output_dir: str,
    frames_per_segment: int,
) -> dict[str, dict[str, Any]]:
    """
    Cut every video in `video_filenames` (each looked up under `videos_dir`) into
    consecutive segments of `frames_per_segment` frames each.

    Returns:
        segment_metadata: {segment_path: {"source_video": origin_video_filename,
        "start_frame": int, "end_frame": int}}.
    """
    os.makedirs(output_dir, exist_ok=True)

    segment_metadata: dict[str, dict[str, Any]] = {}
    for video_filename in video_filenames:
        video_path = os.path.join(videos_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"Video file not found, skipping: {video_path}")
            continue

        segments = split_video_into_segments(
            video_path=video_path,
            output_dir=output_dir,
            frames_per_segment=frames_per_segment,
        )
        for segment in segments:
            segment_metadata[segment.path] = {
                "source_video": video_filename,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
            }

    print(
        f"Produced {len(segment_metadata)} segment(s) from {len(video_filenames)} video(s)."
    )
    return segment_metadata


def build_segment_video_coco(
    video_coco_data: dict[str, Any],
    segment_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Remap the original per-video track annotations (video COCO format, keyed by
    `video_id`/`frame_id`/`track_id`) onto the newly created video segments.

    `segment_metadata` maps segment_path -> {"source_video": ..., "start_frame": ...,
    "end_frame": ...} as produced by `split_video_into_segments`.

    Returns a video COCO dict (with 'videos', 'images', 'annotations') ready to be
    imported with `import_annotations_coco_video_file`.
    """
    video_id_to_filename: dict[int, str] = {
        v["id"]: v["file_name"] for v in video_coco_data.get("videos", [])
    }
    image_id_to_image: dict[int, dict[str, Any]] = {
        img["id"]: img for img in video_coco_data.get("images", [])
    }

    filename_to_segments: dict[str, list[tuple[int, int, str]]] = {}
    segment_path_to_new_video_id: dict[str, int] = {}
    for new_video_id, (segment_path, meta) in enumerate(segment_metadata.items()):
        segment_path_to_new_video_id[segment_path] = new_video_id
        filename_to_segments.setdefault(meta["source_video"], []).append(
            (meta["start_frame"], meta["end_frame"], segment_path)
        )

    annotations_out: list[dict[str, Any]] = []
    new_ann_id = 0
    skipped_missing_track_info = 0
    skipped_out_of_range = 0

    for ann in video_coco_data.get("annotations", []):
        attributes = ann.get("attributes") or {}
        track_id = attributes.get("track_id")
        frame_id = attributes.get("frame_id")
        video_id = ann.get("video_id")

        image = image_id_to_image.get(ann.get("image_id"))
        if video_id is None and image is not None:
            video_id = image.get("video_id")
        if frame_id is None and image is not None:
            frame_id = image.get("frame_id")

        if track_id is None or frame_id is None or video_id is None:
            skipped_missing_track_info += 1
            continue

        origin_filename = video_id_to_filename.get(video_id)
        if origin_filename is None:
            continue

        matched_segment = next(
            (
                seg
                for seg in filename_to_segments.get(origin_filename, [])
                if seg[0] <= frame_id < seg[1]
            ),
            None,
        )
        if matched_segment is None:
            skipped_out_of_range += 1
            continue

        start_frame, _, segment_path = matched_segment
        new_ann: dict[str, Any] = {
            "id": new_ann_id,
            "image_id": new_ann_id,
            "category_id": ann["category_id"],
            "bbox": ann.get("bbox", []),
            "area": ann.get("area", 0.0),
            "iscrowd": ann.get("iscrowd", 0),
            "video_id": segment_path_to_new_video_id[segment_path],
            "attributes": {
                "track_id": track_id,
                "frame_id": frame_id - start_frame,
                "keyframe": True,
            },
        }
        seg = ann.get("segmentation")
        if seg:
            new_ann["segmentation"] = seg

        annotations_out.append(new_ann)
        new_ann_id += 1

    if skipped_missing_track_info:
        print(
            f"Skipped {skipped_missing_track_info} annotation(s) missing "
            f"track/frame information."
        )
    if skipped_out_of_range:
        print(
            f"Skipped {skipped_out_of_range} annotation(s) whose frame fell "
            f"outside any produced segment."
        )

    videos_out = [
        {"id": new_video_id, "file_name": os.path.basename(segment_path)}
        for segment_path, new_video_id in segment_path_to_new_video_id.items()
    ]

    print(
        f"Built segment video COCO: {len(videos_out)} video(s), "
        f"{len(annotations_out)} annotation(s)."
    )
    return {
        "categories": video_coco_data.get("categories", []),
        "videos": videos_out,
        "images": [],
        "annotations": annotations_out,
    }


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

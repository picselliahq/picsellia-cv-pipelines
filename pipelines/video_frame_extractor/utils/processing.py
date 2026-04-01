import logging
import os
from pathlib import Path

import cv2

logger = logging.getLogger("picsellia-engine")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def is_video_file(filepath: str) -> bool:
    return Path(filepath).suffix.lower() in VIDEO_EXTENSIONS


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: int = 0,
) -> list[dict]:
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract one frame every N frames.
        max_frames: Maximum number of frames to extract. 0 means no limit.

    Returns:
        List of dicts with frame metadata (file_name, width, height).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return []

    video_name = Path(video_path).stem
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(
        f"Processing video '{video_name}': {total_frames} frames, {fps:.1f} FPS, "
        f"extracting every {frame_interval} frames"
    )

    extracted = []
    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            cv2.imwrite(frame_path, frame)

            height, width = frame.shape[:2]
            extracted.append(
                {
                    "file_name": frame_filename,
                    "width": width,
                    "height": height,
                    "source_video": Path(video_path).name,
                    "frame_index": frame_idx,
                }
            )

            extracted_count += 1
            if max_frames > 0 and extracted_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {extracted_count} frames from '{video_name}'")
    return extracted


def extract_frames_from_all_videos(
    input_dir: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames_per_video: int = 0,
) -> list[dict]:
    """
    Extract frames from all video files in a directory.

    Args:
        input_dir: Directory containing video files.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract one frame every N frames.
        max_frames_per_video: Max frames per video. 0 means no limit.

    Returns:
        List of dicts with frame metadata (file_name, width, height).
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(os.listdir(input_dir))
    video_files = [f for f in all_files if is_video_file(f)]

    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return []

    logger.info(f"Found {len(video_files)} video(s) to process")

    all_frames = []
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        frames = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_interval=frame_interval,
            max_frames=max_frames_per_video,
        )
        all_frames.extend(frames)

    logger.info(f"Total frames extracted: {len(all_frames)}")
    return all_frames

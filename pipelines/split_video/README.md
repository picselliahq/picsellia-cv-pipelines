# Split Video Pipeline

**Cut video dataset versions into shorter video segments and create a new video dataset version.**

This Picsellia pipeline downloads video assets from a dataset version, cuts each video into consecutive segments of a fixed number of frames using OpenCV, uploads the segments to a datalake, and creates a new video dataset version on the same parent dataset — carrying over any existing track annotations, remapped onto the segment that now contains each keyframe.

## What You'll Get

After running this pipeline, you'll have:
- A new video dataset version containing all the video segments
- Segments uploaded to the datalake with the tag `video_segment` and `source_video`/`source_video_tags` custom metadata
- If the source videos had track annotations, they are preserved: each keyframe is reassigned to the segment that now contains it, with its frame index shifted to be relative to that segment. A track that spans a cut point is split into two independent tracks, one per segment.

---

## Inputs Reference

Inputs are configured when launching the processing job from the Picsellia platform.

### `output_dataset_version_name`
**What it does**: Name of the new video dataset version to create on the same parent dataset as the input video dataset version.

**Type**: Text
**Required**: Yes

**Example**: `segments_v1`, `10s_clips`

If a version with this name already exists, the pipeline appends a timestamp to avoid conflicts.

---

### `datalake`
**What it does**: The datalake where the video segments will be uploaded before being added to the new dataset version.

**Type**: Datalake
**Required**: Yes

Select the datalake from your Picsellia workspace.

---

### `frames_per_segment`
**What it does**: Number of frames in each output video segment. Every source video is cut into consecutive segments of this many frames; the final segment of a video may be shorter if the video's frame count isn't an exact multiple of this value.

**Type**: Number
**Required**: Yes

**Example**: `100`, `300`

---

## Parameters Reference

This pipeline has no configurable parameters.

---

## How It Works

The pipeline runs four steps in sequence, handling every VIDEO-type asset in the dataset version:

### Step 1 — `download_videos`
Lists all assets in the input dataset version and filters for VIDEO-type assets. If specific asset IDs are selected in the processing job, only those assets are used. Downloads the video files locally, reads each video's tags (`asset.get_tags()`), then exports their COCO annotations using the video COCO format (`export_video=True`), which includes track-based annotations with a `frame_id`/`track_id` per keyframe.

### Step 2 — `split_videos`
For every downloaded video, opens it with OpenCV (`cv2.VideoCapture`) and reads its FPS, width and height. Writes consecutive segments of `frames_per_segment` frames each with `cv2.VideoWriter`, named `{video_stem}_part_{index:03d}.mp4`. Each segment's absolute `[start_frame, end_frame)` range in its source video is tracked for annotation remapping. Videos are processed independently, so a dataset with several videos of different lengths produces a different number of segments per video.

### Step 3 — `upload_segments_and_create_dataset`
Uploads all video segments (from every source video) to the selected datalake with:
- Tag: `video_segment`
- Custom metadata: `{"source_video": "<origin_video_filename>", "source_video_tags": ["<tag_name>", ...]}` per segment

Then creates a new dataset version on the same parent dataset and adds the uploaded segments to it.

### Step 4 — `upload_annotations`
If the source video(s) had annotations, remaps each original keyframe (identified by `video_id`/`frame_id`/`track_id`) onto the segment whose `[start_frame, end_frame)` range now contains it, shifting `frame_id` to be relative to that segment. Detects the inference type from the remapped annotations and imports them via `import_annotations_coco_video_file`. Only Object Detection and Segmentation are supported for video-track import; other types are skipped with a warning. If there are no source annotations, this step is a no-op.

---

## Input Dataset Requirements

- The input dataset version must contain **VIDEO-type assets**. Any number of videos is supported — each is split independently.
- Annotations are optional. If present, they must be track-based video annotations (Object Detection or Segmentation) — this is what `export_video=True` produces. Other annotation types (Mask, Keypoint, Classification) cannot be re-imported as video tracks and are skipped.

---

## Segment Naming Convention

Output segments follow this naming pattern:

```
{video_stem}_part_{segment_index:03d}.mp4
```

**Example** — for a video named `highway_cam.mp4` split into 100-frame segments:
```
highway_cam_part_000.mp4
highway_cam_part_001.mp4
highway_cam_part_002.mp4
...
```

---

## Datalake Metadata

Every uploaded segment carries:

| Field | Value |
|-------|-------|
| Tag | `video_segment` |
| Custom metadata | `{"source_video": "highway_cam.mp4", "source_video_tags": ["night", "downtown"]}` |

This allows filtering segments by origin video, or by the tags carried over from the origin video, in the Picsellia datalake.

---

## Quick Start

| Input | Example value |
|-------|---------------|
| `output_dataset_version_name` | `segments_v1` |
| `datalake` | *(select your datalake)* |
| `frames_per_segment` | `100` |

1. Create a processing job on a dataset version that contains one or more VIDEO assets.
2. Set the inputs above.
3. Run — video segments from every source video appear in the new dataset version.

---

## Troubleshooting

### No video assets found
The pipeline raises an error if no VIDEO-type assets are present in the selected dataset version (or in the selected asset IDs). Ensure the source dataset version contains video files.

### Output version name conflict
If a dataset version with `output_dataset_version_name` already exists on the parent dataset, the pipeline automatically appends a timestamp (e.g., `segments_v1_1719830412.345`) and continues without failing.

### Annotations not imported
If the source video annotations aren't Object Detection or Segmentation tracks (e.g. Mask, Keypoint, Classification), or no keyframe falls within any produced segment, the pipeline logs a message and skips the import. The new dataset version is still created with the video segments.

---

**Pipeline Version**: 1.0
**Type**: Dataset Version Creation
**Supported Input Types**: VIDEO assets
**Supported Output Types**: VIDEO (Object Detection and Segmentation track annotations)

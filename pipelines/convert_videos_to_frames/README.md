# Convert Videos to Frames Pipeline

**Split annotated video dataset versions into individual frames and create a new image dataset version.**

This Picsellia pipeline downloads video assets from a dataset version, extracts every frame using OpenCV, uploads the frames to a datalake, and creates a new image dataset version on the same parent dataset — with annotations converted from the video COCO format (track-based) to the standard COCO image format (per-frame).

## What You'll Get

After running this pipeline, you'll have:
- A new image dataset version containing all extracted frames
- Frames uploaded to the datalake with the tag `video_frames` and `origin_video` custom metadata
- Annotations converted from video tracks to standard per-frame COCO format
- Inference type automatically detected from the source annotations (Object Detection, Segmentation, Mask, Keypoint, or Classification)

---

## Inputs Reference

Inputs are configured when launching the processing job from the Picsellia platform.

### `output_dataset_version_name`
**What it does**: Name of the new image dataset version to create on the same parent dataset as the input video dataset version.

**Type**: Text
**Required**: Yes

**Example**: `frames_v1`, `training_frames`

If a version with this name already exists, the pipeline appends a timestamp to avoid conflicts.

---

### `datalake`
**What it does**: The datalake where the extracted frames will be uploaded before being added to the new dataset version.

**Type**: Datalake
**Required**: Yes

Select the datalake from your Picsellia workspace.

---

## Parameters Reference

This pipeline has no configurable parameters.

---

## How It Works

The pipeline runs four steps in sequence:

### Step 1 — `download_videos`
Lists all assets in the input dataset version and filters for VIDEO-type assets. If specific asset IDs are selected in the processing job, only those assets are used. Downloads the video files locally, then exports their COCO annotations using the video COCO format (`export_video=True`), which includes track-based annotations with `frame_id` per annotated frame.

### Step 2 — `extract_frames`
Opens each video with OpenCV (`cv2.VideoCapture`) and reads every frame sequentially. Each frame is saved as a JPEG named `{video_stem}_frame_{frame_index:06d}.jpg`. A standard COCO `images` list is built from all extracted frames, and annotated frames are identified by matching `(video_id, frame_id)` pairs from the video COCO export. Video track annotations are converted to standard per-frame COCO annotations with remapped `image_id` values.

### Step 3 — `upload_frames_and_create_dataset`
Uploads all extracted frames to the selected datalake with:
- Tag: `video_frames`
- Custom metadata: `{"origin_video": "<origin_video_filename>"}` per frame

Then creates a new dataset version on the same parent dataset and adds the uploaded frames to it.

### Step 4 — `upload_annotations`
Detects the inference type from the converted COCO file (based on annotation structure), sets the type on the new dataset version, and imports the frame annotations via `import_annotations_coco_file`.

---

## Input Dataset Requirements

- The input dataset version must contain **VIDEO-type assets**.
- Videos should have COCO annotations exported in video format (track-based). If no annotations exist, the new dataset version is created without annotations.
- Supported annotation types for the output: Object Detection, Segmentation, Mask, Keypoint, Classification.

---

## Frame Naming Convention

Extracted frames follow this naming pattern:

```
{video_stem}_frame_{frame_index:06d}.jpg
```

**Example** — for a video named `highway_cam.mp4`:
```
highway_cam_frame_000000.jpg
highway_cam_frame_000001.jpg
highway_cam_frame_000002.jpg
...
```

---

## Datalake Metadata

Every uploaded frame carries:

| Field | Value |
|-------|-------|
| Tag | `video_frames` |
| Custom metadata | `{"origin_video": "highway_cam.mp4"}` |

This allows filtering frames by origin video in the Picsellia datalake.

---

## Quick Start

| Input | Example value |
|-------|---------------|
| `output_dataset_version_name` | `frames_v1` |
| `datalake` | *(select your datalake)* |

1. Create a processing job on a dataset version that contains VIDEO assets.
2. Set the two inputs above.
3. Run — frames and converted annotations appear in the new dataset version.

---

## Troubleshooting

### No video assets found
The pipeline raises an error if no VIDEO-type assets are present in the selected dataset version (or in the selected asset IDs). Ensure the source dataset version contains video files.

### Annotations not imported
If the source videos have no COCO annotations, the pipeline skips the annotation import step and logs a message. The new dataset version is still created with the extracted frames.

### Output version name conflict
If a dataset version with `output_dataset_version_name` already exists on the parent dataset, the pipeline automatically appends a timestamp (e.g., `frames_v1_1719830412.345`) and continues without failing.

---

**Pipeline Version**: 1.0
**Type**: Dataset Version Creation
**Supported Input Types**: VIDEO assets
**Supported Output Types**: Object Detection, Segmentation, Mask, Keypoint, Classification

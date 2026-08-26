# SAM3_polygons_from_box Pipeline

**Convert bounding box annotations into precise polygon segmentation annotations using SAM-3, prompted with each existing box.**

Unlike `SAM3_Bbox` and `SAM3_polygons` (which detect new objects from a text/box prompt), this pipeline starts from a dataset **already annotated with bounding boxes**. For every existing box, it prompts SAM-3 with that exact box as a spatial prompt to get a tight segmentation mask, then converts that mask into a polygon.

The output is a **new dataset version** (type `SEGMENTATION`) forked from the input dataset - the same images are reused (no re-upload), only the annotations change from boxes to polygons. Each polygon keeps the **same label/category** as the box it was generated from.

## Required Inputs

- `output_dataset_version_name`: name of the new dataset version to create.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `threshold` | `0.3` | Confidence threshold applied to SAM-3's output for each box prompt. |
| `mask_threshold` | `0.5` | Controls how tight/loose the resulting polygon is. |
| `min_area` | `10.0` | Minimum polygon area (px²); smaller masks are discarded. |
| `fallback_to_bbox_polygon` | `true` | If SAM-3 finds no valid mask for a box, use a rectangular polygon matching that box instead of dropping it, so every input box still produces an output polygon. |
| `annotation_mode` | `"replace"` | `"keep"`, `"replace"` or `"concatenate"` - how to handle annotations already present on the output dataset version. |

## Requirements

- Input dataset version must be of type `OBJECT_DETECTION` (or `NOT_CONFIGURED`) and contain bounding box annotations.

## Example

```toml
[inputs]
output_dataset_version_name = "polygons-from-boxes"

[parameters]
threshold = 0.3
mask_threshold = 0.5
min_area = 20.0
fallback_to_bbox_polygon = true
```

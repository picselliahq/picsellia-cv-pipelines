# SAM3_Bbox Multi-Class Detection

This pipeline now supports **multi-class object detection** using comma-separated text prompts.

## Overview

SAM-3 is inherently a single-class segmentation model. To enable multi-class detection, this pipeline:

1. **Parses comma-separated text prompts** (e.g., `"car,person,bicycle"`)
2. **Runs SAM-3 inference separately** for each class
3. **Deduplicates overlapping detections** across different classes
4. **Assigns appropriate labels** to each detected object

## Usage

### Single-Class Detection (Original Behavior)

```toml
[parameters]
text_prompt = "car"
threshold = 0.3
mask_threshold = 0.5
```

### Multi-Class Detection (New Feature)

```toml
[parameters]
text_prompt = "car,person,bicycle"
threshold = 0.3
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
```

## Deduplication Strategy

When detecting multiple classes, the same object might be detected multiple times with different labels. The pipeline handles this using two metrics:

### 1. IoU-Based Deduplication
- Detects similar-sized overlapping objects
- If `IoU > iou_threshold`, masks are considered duplicates
- Default: `iou_threshold = 0.5`

### 2. Containment-Based Deduplication
- Detects when one mask is nested inside another
- Handles scenarios like a small object in front of a larger one
- If `containment_ratio > containment_threshold`, masks are duplicates
- Default: `containment_threshold = 0.8`

### Deduplication Strategy Options

**`keep_smaller`** (Default - Recommended)
- Prioritizes smaller, more precise masks
- Best when SAM-3 tends to over-segment
- Example: Small "person" mask kept over large "car" mask when person is in foreground

**`keep_larger`**
- Prioritizes larger, more complete masks
- Best when you want full object coverage
- Example: Large "car" mask kept over small partial detections

## Parameters Reference

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `text_prompt` | string | Single or comma-separated class names | `"car,person,bicycle"` |
| `threshold` | float | Detection confidence threshold (0-1) | `0.3` |
| `mask_threshold` | float | Mask generation threshold (0-1) | `0.5` |

### Multi-Class Deduplication Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iou_threshold` | float | `0.5` | IoU threshold for overlap detection (0-1) |
| `containment_threshold` | float | `0.8` | Threshold for nested mask detection (0-1) |
| `deduplication_strategy` | string | `"keep_smaller"` | Strategy: `"keep_smaller"` or `"keep_larger"` |

### Other Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_area` | float | `50.0` | Minimum mask area in pixels |
| `max_overlap_ratio` | float | `0.3` | Same-class overlap removal threshold |
| `box_prompt` | list | None | Bounding box constraint `[x1, y1, x2, y2]` |
| `label_name` | string | `"object"` | Fallback label for box_prompt only |

## Examples

### Example 1: Street Scene Detection

```toml
[parameters]
text_prompt = "car,person,bicycle,traffic light"
threshold = 0.35
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
min_area = 100.0
```

### Example 2: Warehouse Inventory

```toml
[parameters]
text_prompt = "box,pallet,forklift"
threshold = 0.4
mask_threshold = 0.6
iou_threshold = 0.6
containment_threshold = 0.75
deduplication_strategy = "keep_larger"
min_area = 200.0
```

### Example 3: Waste Detection

```toml
[parameters]
text_prompt = "plastic bottle,paper,metal can,glass"
threshold = 0.3
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
min_area = 50.0
```

## How It Works Internally

1. **Parse text prompts**: Split `"car,person,bicycle"` → `["car", "person", "bicycle"]`

2. **Create categories**: Each prompt becomes a label in the dataset

3. **Per-image processing**:
   - For each text prompt:
     - Run SAM-3 inference
     - Collect all detected masks with class information
   - Deduplicate across all classes using IoU + containment metrics
   - Convert to COCO annotations

4. **Deduplication logic**:
   ```
   For each detection (sorted by strategy):
     For each already-kept detection:
       Calculate IoU and containment
       If IoU > threshold OR containment > threshold:
         Mark as duplicate, skip
     If not duplicate:
       Keep detection
   ```

## Troubleshooting

### Too many duplicate detections
- Increase `iou_threshold` (e.g., `0.7`)
- Increase `containment_threshold` (e.g., `0.9`)

### Missing valid detections
- Decrease `iou_threshold` (e.g., `0.3`)
- Decrease `containment_threshold` (e.g., `0.6`)
- Adjust `deduplication_strategy`

### Wrong class labels on objects
- Try switching `deduplication_strategy`
- Adjust `containment_threshold` for better nested object handling

### Performance issues
- Reduce number of prompts
- Increase `threshold` to reduce detections per class
- Increase `min_area` to filter small masks early

## Technical Details

- **Language**: Each text prompt runs SAM-3 inference independently
- **Deduplication**: Happens per-image after all prompts are processed
- **Output**: Standard COCO format with multiple categories
- **Polygon validation**: Uses Shapely for robust geometry operations
- **GPU support**: Automatic CUDA detection and usage

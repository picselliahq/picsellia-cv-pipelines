# Dataset Tiler Pipeline

**Split large images into smaller tiles while preserving annotations.**

This Picsellia pipeline divides large images into smaller, uniformly-sized tiles, automatically adjusting bounding boxes and segmentation masks to fit the new tile boundaries. Perfect for satellite imagery, aerial photos, gigapixel images, or any high-resolution datasets where objects are small relative to image size.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A new dataset version with tiled images
- ✅ Annotations correctly split and adjusted for each tile
- ✅ Configurable tile size and overlap
- ✅ Optional filtering of tiny or partial annotations
- ✅ Support for Object Detection, Segmentation, and Classification datasets

---

## Inputs vs Parameters

This pipeline uses two distinct configuration mechanisms:

| Mechanism | Set when | Examples |
|-----------|----------|---------|
| **Inputs** | Launching the job on Picsellia | target dataset, datalake, tile size, tiling mode |
| **Parameters** | Configuring the processing | overlap ratios, annotation filters, padding |

---

## 📥 Inputs Reference

Inputs are configured when launching the processing job from the Picsellia platform.

### `target_version_name`
**What it does**: Name of the output dataset version to create (or reuse if it already exists).

**Type**: Text
**Required**: Yes

**Example**: `tiled_v1`, `640x640_tiles`

---

### `datalake`
**What it does**: The datalake where tiled images will be uploaded.

**Type**: Datalake
**Required**: Yes

Select the datalake from your Picsellia workspace.

---

### `data_tag`
**What it does**: Tag applied to all tiled images uploaded to the datalake.

**Type**: Text
**Required**: Yes

**Example**: `tiled_data`, `satellite_tiles_640`

---

### `tile_height`
**What it does**: Height of each tile in pixels.

**Type**: Number
**Required**: Yes
**Recommended values**: `416`, `512`, `640`, `1024`

**How to choose**:
- Match your model's input size (e.g., 640 for YOLOv8)
- Larger tiles = fewer tiles, less overlap issues
- Smaller tiles = more tiles, better for small objects

---

### `tile_width`
**What it does**: Width of each tile in pixels.

**Type**: Number
**Required**: Yes
**Recommended values**: `416`, `512`, `640`, `1024`

**Note**: Most detection models work best with square tiles.

---

### `tiling_mode`
**What it does**: How to handle edges when image dimensions don't divide evenly by tile size.

**Type**: Text
**Required**: Yes
**Options**: `CONSTANT`, `DROP`, `REFLECT`, `EDGE`, `WRAP`

**How each mode works**:

| Mode | Behaviour | Best For |
|------|-----------|----------|
| `CONSTANT` | Pads edges with `padding_color_value` | Most use cases, clean edges |
| `DROP` | Drops edge tiles that don't fit completely | When edge regions are unimportant |
| `REFLECT` | Reflects image at boundaries | Natural images, avoid hard edges |
| `EDGE` | Repeats edge pixels | Extend existing content smoothly |
| `WRAP` | Wraps around to opposite edge | Seamless textures (rare in CV) |

---

## 📋 Parameters Reference

Parameters are set when configuring the processing on Picsellia.

### 🔵 Overlap Parameters

#### `overlap_height_ratio`
**What it does**: Vertical overlap between adjacent tiles as a fraction of tile height.

**Type**: Float
**Range**: `0.0` to `0.99`
**Default**: `0.1`
**Recommended range**: `0.1` to `0.3`

**How it works**:
```
overlap_height_ratio = 0.0  →  No overlap (tiles are adjacent)
overlap_height_ratio = 0.2  →  20% overlap (tiles share 128px if tile_height=640)
overlap_height_ratio = 0.5  →  50% overlap (tiles share 320px)
```

**Why use overlap**:
- ✅ Objects at tile edges appear in multiple tiles
- ✅ Reduces risk of cutting objects in half
- ✅ Better coverage during training
- ❌ Creates more tiles (increases processing time)

**When to adjust**:
- **Dense objects near edges**: Use 0.2-0.3
- **Sparse scenes**: Use 0.1 or 0.0
- **Speed is priority**: Use 0.0

---

#### `overlap_width_ratio`
**What it does**: Horizontal overlap between adjacent tiles as a fraction of tile width.

**Type**: Float
**Range**: `0.0` to `0.99`
**Default**: `0.1`
**Recommended range**: `0.1` to `0.3`

**Tip**: Usually set equal to `overlap_height_ratio` for uniform coverage.

---

### 🔵 Annotation Filtering Parameters

#### `min_annotation_area_ratio`
**What it does**: Minimum annotation area as a fraction of the original annotation area.

**Type**: Float
**Range**: `0.0` to `0.99`
**Default**: `0.0` (keep all)

**How it works**:
```
Original annotation area: 1000 px²
After tiling, partial annotation: 200 px²
Area ratio: 200/1000 = 0.2 (20% of original)

If min_annotation_area_ratio = 0.3 → Annotation is DISCARDED
If min_annotation_area_ratio = 0.1 → Annotation is KEPT
```

---

#### `min_annotation_width`
**What it does**: Minimum annotation width in pixels after tiling.

**Type**: Integer
**Range**: `>= 0`
**Default**: `0` (no filtering)

---

#### `min_annotation_height`
**What it does**: Minimum annotation height in pixels after tiling.

**Type**: Integer
**Range**: `>= 0`
**Default**: `0` (no filtering)

---

### 🔵 Padding Parameter

#### `padding_color_value`
**What it does**: Color value for padding when `tiling_mode = CONSTANT`.

**Type**: Integer
**Range**: `0` to `255`
**Default**: `114` (neutral gray)

**Common values**:
- `0` = Black
- `114` = Gray (YOLO standard)
- `127` = Mid gray
- `255` = White

---

### 🔵 Validation Parameter

#### `fix_annotation`
**What it does**: Attempt to fix broken annotations during validation.

**Type**: Boolean
**Default**: `true`

**What it fixes**:
- Invalid bounding boxes (negative coordinates, out of bounds)
- Malformed polygons
- Duplicate annotations

---

## Quick Start Guide

### 🎯 Basic Tiling (640×640)

**Inputs** (set when launching the job):
| Input | Value |
|-------|-------|
| `target_version_name` | `tiled_v1` |
| `datalake` | *(select your datalake)* |
| `data_tag` | `tiled_data` |
| `tile_height` | `640` |
| `tile_width` | `640` |
| `tiling_mode` | `CONSTANT` |

**Parameters**:
```toml
[parameters]
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0
```

---

### 🎯 Tiling with Overlap

**Inputs**: `tile_height = 640`, `tile_width = 640`, `tiling_mode = CONSTANT`

**Parameters**:
```toml
[parameters]
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2
```

20% overlap ensures objects at tile boundaries appear in multiple tiles.

---

### 🎯 Tiling for Small Object Detection

**Inputs**: `tile_height = 640`, `tile_width = 640`, `tiling_mode = CONSTANT`

**Parameters**:
```toml
[parameters]
overlap_height_ratio = 0.15
overlap_width_ratio = 0.15
min_annotation_area_ratio = 0.1
min_annotation_width = 10
min_annotation_height = 10
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Satellite Imagery (4000×4000 → 640×640)

**Inputs**: `tile_height = 640`, `tile_width = 640`, `tiling_mode = CONSTANT`, `data_tag = satellite_tiles`

**Parameters**:
```toml
[parameters]
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2
min_annotation_area_ratio = 0.15
min_annotation_width = 10
min_annotation_height = 10
padding_color_value = 0
fix_annotation = true
```

~49 tiles per 4000×4000 image. Black padding for space imagery.

---

### Example 2: Aerial Drone Footage (High Resolution)

**Inputs**: `tile_height = 1024`, `tile_width = 1024`, `tiling_mode = CONSTANT`

**Parameters**:
```toml
[parameters]
overlap_height_ratio = 0.25
overlap_width_ratio = 0.25
min_annotation_area_ratio = 0.2
min_annotation_width = 20
min_annotation_height = 20
padding_color_value = 114
```

---

### Example 3: Medical Imaging (Whole Slide Images)

**Inputs**: `tile_height = 512`, `tile_width = 512`, `tiling_mode = DROP`

**Parameters**:
```toml
[parameters]
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0
min_annotation_area_ratio = 0.0
fix_annotation = true
```

Drop edge tiles (slide edges are often empty).

---

### Example 4: Fast Processing / No Overlap

**Inputs**: `tile_height = 640`, `tile_width = 640`, `tiling_mode = DROP`

**Parameters**:
```toml
[parameters]
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0
min_annotation_area_ratio = 0.0
```

No overlap = minimal tiles, ~4× faster than 25% overlap.

---

## 🔧 Parameter Tuning Workflow

### Step 1: Calculate Tile Count

```
Formula (no overlap):
tiles_per_image = ceil(image_width / tile_width) × ceil(image_height / tile_height)

Example:
Image: 3840 × 2160, Tiles: 640 × 640
tiles_per_image = ceil(3840/640) × ceil(2160/640) = 6 × 4 = 24 tiles

With 20% overlap:
Effective tile step = 640 × (1 - 0.2) = 512
tiles_per_image = ceil(3840/512) × ceil(2160/512) = 8 × 5 = 40 tiles
```

### Step 2: Test on Sample Images

Run on 5-10 images first with conservative settings:

**Inputs**: `tile_height = 640`, `tile_width = 640`, `tiling_mode = CONSTANT`

**Parameters**:
```toml
overlap_height_ratio = 0.15
overlap_width_ratio = 0.15
min_annotation_area_ratio = 0.1
```

### Step 3: Adjust Based on Results

| Observation | Action |
|-------------|--------|
| Objects cut at edges | Increase overlap (try 0.2-0.3) |
| Too many tiles | Reduce overlap or increase tile size (input) |
| Many tiny fragments | Increase `min_annotation_area_ratio` |
| Missing partial objects | Decrease filtering parameters |
| Edge padding looks bad | Change `tiling_mode` input (try `REFLECT` or `DROP`) |

---

## ❓ Troubleshooting Guide

### Issue: Too Many Tiles Created
1. Reduce overlap ratios
2. Increase tile size (change `tile_height`/`tile_width` inputs)
3. Use `tiling_mode = DROP` to skip edge tiles

### Issue: Objects Cut in Half
1. Increase overlap to 0.2-0.3
2. Consider larger tile size input

### Issue: Many Tiny Useless Annotations
1. Increase `min_annotation_area_ratio` to 0.2-0.3
2. Set `min_annotation_width` and `min_annotation_height` (e.g., 15-20)

### Issue: Important Partial Annotations Removed
1. Decrease `min_annotation_area_ratio` (try 0.05-0.1)
2. Reduce `min_annotation_width`/`min_annotation_height`

### Issue: Pipeline Fails on Invalid Annotations
1. Ensure `fix_annotation = true`
2. Review original dataset quality

### Issue: Edge Tiles Look Strange
Change the `tiling_mode` input:
- `REFLECT` — natural boundaries
- `EDGE` — extend edges
- `DROP` — skip edge tiles entirely

---

## 💡 Best Practices

1. **Match tile size to your model** — set `tile_height`/`tile_width` inputs to your model's expected input size (e.g., 640 for YOLOv8)
2. **Use overlap for detection tasks** — `overlap_height_ratio = overlap_width_ratio = 0.15` prevents missing objects at boundaries
3. **Filter intelligently** — `min_annotation_area_ratio = 0.15` with `min_annotation_width/height = 10` removes noise while preserving important objects
4. **Test on samples first** — always run on 5-10 images before the full dataset
5. **Preserve the original dataset** — the tiler creates a new dataset version; originals are untouched

---

## 📊 Common Use Cases Summary

| Use Case | `tile_height/width` | `tiling_mode` | Overlap | `min_area_ratio` |
|----------|---------------------|---------------|---------|-----------------|
| Satellite imagery | 640 | CONSTANT | 0.2 | 0.15 |
| Aerial photos | 1024 | CONSTANT | 0.25 | 0.2 |
| Medical images | 512 | DROP | 0.0 | 0.0 |
| Documents | 640 | CONSTANT | 0.1 | 0.3 |
| Retail shelves | 640 | CONSTANT | 0.15 | 0.25 |
| Fast processing | any | DROP | 0.0 | 0.0 |

---

**Pipeline Version**: 1.0
**Type**: Dataset Version Creation
**Supported Types**: Object Detection, Segmentation, Classification

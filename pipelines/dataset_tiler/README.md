# Dataset Tiler Pipeline

**Split large images into smaller tiles while preserving annotations.**

This Picsellia pipeline divides large images into smaller, uniformly-sized tiles, automatically adjusting bounding boxes and segmentation masks to fit the new tile boundaries. Perfect for satellite imagery, aerial photos, gigapixel images, or any high-resolution datasets where objects are small relative to image size.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A new dataset with tiled images
- ✅ Annotations correctly split and adjusted for each tile
- ✅ Configurable tile size and overlap
- ✅ Optional filtering of tiny or partial annotations
- ✅ Support for Object Detection, Segmentation, and Classification datasets

## Quick Start Guide

### 🎯 Basic Tiling

**Goal**: Split images into 640x640 tiles with no overlap.

```toml
[parameters]
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0
```

Simple grid-based tiling of images.

### 🎯 Tiling with Overlap

**Goal**: Create overlapping tiles to avoid cutting objects at edges.

```toml
[parameters]
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2
```

20% overlap ensures objects at tile boundaries appear in multiple tiles.

### 🎯 Tiling for Small Object Detection

**Goal**: Tile large images to improve small object detection in training.

```toml
[parameters]
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.15
overlap_width_ratio = 0.15
min_annotation_area_ratio = 0.1
min_annotation_width = 10
min_annotation_height = 10
```

Filters out tiny partial annotations while preserving meaningful objects.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Tiling Parameters

#### `tile_height`
**What it does**: Height of each tile in pixels.

**Type**: Integer
**Range**: `> 0`
**Default**: `640`
**Recommended values**: `416`, `512`, `640`, `1024`

**How to choose**:
- Match your model's input size (e.g., 640 for YOLOv8)
- Larger tiles = fewer tiles, less overlap issues
- Smaller tiles = more tiles, better for small objects

**Example**:
```toml
# Standard YOLO size
tile_height = 640

# High-resolution tiling
tile_height = 1024

# Fast inference size
tile_height = 416
```

---

#### `tile_width`
**What it does**: Width of each tile in pixels.

**Type**: Integer
**Range**: `> 0`
**Default**: `640`
**Recommended values**: `416`, `512`, `640`, `1024`

**Example**:
```toml
tile_width = 640

# Non-square tiles
tile_width = 800
tile_height = 600
```

**Note**: Most detection models work best with square tiles.

---

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

**Visual understanding**:
```
No overlap (0.0):
┌───┐┌───┐┌───┐
│ 1 ││ 2 ││ 3 │
└───┘└───┘└───┘

20% overlap (0.2):
┌───┐
│ 1 │
└─┬─┘
  │ 2 │
  └─┬─┘
    │ 3 │
    └───┘
```

**Why use overlap**:
- ✅ Objects at tile edges appear in multiple tiles
- ✅ Reduces risk of cutting objects in half
- ✅ Better coverage during training
- ❌ Creates more tiles (increases processing time)
- ❌ Same object may appear in multiple tiles

**When to adjust**:
- **Dense objects near edges**: Use 0.2-0.3
- **Sparse scenes**: Use 0.1 or 0.0
- **Critical not to miss objects**: Use 0.25-0.4
- **Speed is priority**: Use 0.0

**Example**:
```toml
# Standard overlap
overlap_height_ratio = 0.15

# No overlap for speed
overlap_height_ratio = 0.0

# High overlap to preserve objects
overlap_height_ratio = 0.3
```

---

#### `overlap_width_ratio`
**What it does**: Horizontal overlap between adjacent tiles as a fraction of tile width.

**Type**: Float
**Range**: `0.0` to `0.99`
**Default**: `0.1`
**Recommended range**: `0.1` to `0.3`

**Example**:
```toml
# Symmetric overlap
overlap_width_ratio = 0.2
overlap_height_ratio = 0.2

# Asymmetric overlap (rare)
overlap_width_ratio = 0.3
overlap_height_ratio = 0.1
```

**Tip**: Usually set equal to `overlap_height_ratio` for uniform coverage.

---

### 🔵 Annotation Filtering Parameters

These parameters filter out annotations that are too small or too fragmented after tiling.

#### `min_annotation_area_ratio`
**What it does**: Minimum annotation area as a fraction of the original annotation area.

**Type**: Float
**Range**: `0.0` to `0.99`
**Default**: `0.0` (keep all)
**Recommended range**: `0.05` to `0.3`

**How it works**:
```
Original annotation area: 1000 px²
After tiling, partial annotation: 200 px²
Area ratio: 200/1000 = 0.2 (20% of original)

If min_annotation_area_ratio = 0.3 → Annotation is DISCARDED
If min_annotation_area_ratio = 0.1 → Annotation is KEPT
```

**Why use this**:
- Filter out tiny fragments of objects cut by tile boundaries
- Remove annotations that are mostly outside the tile
- Keep only meaningful portions of objects

**When to adjust**:
- **Keep most partial objects**: Use 0.0-0.1
- **Only keep substantial portions**: Use 0.3-0.5
- **Very strict filtering**: Use 0.5+

**Example**:
```toml
# Keep any fragment
min_annotation_area_ratio = 0.0

# Keep at least 20% of original object
min_annotation_area_ratio = 0.2

# Keep only large portions
min_annotation_area_ratio = 0.5
```

---

#### `min_annotation_width`
**What it does**: Minimum annotation width in pixels after tiling.

**Type**: Integer
**Range**: `>= 0`
**Default**: `0` (no filtering)

**Why use this**: Filter out annotations that become too thin after tiling.

**Example**:
```toml
# Remove annotations narrower than 10 pixels
min_annotation_width = 10

# Remove very thin annotations
min_annotation_width = 20
```

**Use case**: Satellite imagery where thin line features might create noise.

---

#### `min_annotation_height`
**What it does**: Minimum annotation height in pixels after tiling.

**Type**: Integer
**Range**: `>= 0`
**Default**: `0` (no filtering)

**Example**:
```toml
# Remove annotations shorter than 10 pixels
min_annotation_height = 10

# Combined with width for minimum size
min_annotation_width = 15
min_annotation_height = 15
```

---

### 🔵 Tiling Mode and Padding

#### `tiling_mode`
**What it does**: How to handle edges when image dimensions don't divide evenly by tile size.

**Type**: String (Enum)
**Default**: `"constant"`
**Options**: `"constant"`, `"drop"`, `"reflect"`, `"edge"`, `"wrap"`

**How each mode works**:

**`"constant"` (Recommended Default)**
- Pads edges with constant color value
- Use `padding_color_value` to set the color

**Example**: Image is 1500x1000, tiles are 640x640
```
Last column/row of tiles will be padded to reach 640x640
Padding color: Gray (114) by default
```

**`"drop"`**
- Drops edge tiles that don't fit completely
- No padding, loses edge regions

**`"reflect"`**
- Reflects image at boundaries
- Creates mirror effect

**`"edge"`**
- Repeats edge pixels
- Extends border colors

**`"wrap"`**
- Wraps around to opposite edge
- Creates circular tiling

**When to use each**:

| Mode | Best For |
|------|----------|
| `"constant"` | Most use cases, clean edges |
| `"drop"` | When edge regions are unimportant |
| `"reflect"` | Natural images, avoid hard edges |
| `"edge"` | Extend existing content smoothly |
| `"wrap"` | Seamless textures (rare in CV) |

**Example**:
```toml
# Default - pad with gray
tiling_mode = "constant"
padding_color_value = 114

# Drop incomplete tiles
tiling_mode = "drop"

# Reflect for natural boundaries
tiling_mode = "reflect"
```

---

#### `padding_color_value`
**What it does**: Color value for padding when `tiling_mode = "constant"`.

**Type**: Integer
**Range**: `0` to `255`
**Default**: `114` (neutral gray)

**Common values**:
- `0` = Black
- `114` = Gray (YOLO standard)
- `127` = Mid gray
- `255` = White

**Example**:
```toml
# Standard gray padding
tiling_mode = "constant"
padding_color_value = 114

# Black padding
padding_color_value = 0

# White padding
padding_color_value = 255
```

---

### 🔵 Utility Parameters

#### `fix_annotation`
**What it does**: Attempt to fix broken annotations during validation.

**Type**: Boolean
**Default**: `true`

**What it fixes**:
- Invalid bounding boxes (negative coordinates, out of bounds)
- Malformed polygons
- Duplicate annotations

**Example**:
```toml
# Fix issues automatically (recommended)
fix_annotation = true

# Strict mode - fail on any invalid annotation
fix_annotation = false
```

---

#### `datalake`
**What it does**: Name of the datalake to store tiled images.

**Type**: String
**Default**: `"default"`

**Example**:
```toml
datalake = "default"

# Custom datalake
datalake = "aerial-imagery"
```

---

#### `data_tag`
**What it does**: Tag to apply to tiled images.

**Type**: String
**Default**: `"tiled_data"`

**Example**:
```toml
data_tag = "tiled_data"

# Custom tag for organization
data_tag = "satellite_tiles_640"
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Satellite Imagery (4000x4000 → 640x640)

**Goal**: Tile large satellite images for object detection.

```toml
[parameters]
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2
min_annotation_area_ratio = 0.15
min_annotation_width = 10
min_annotation_height = 10
tiling_mode = "constant"
padding_color_value = 0
fix_annotation = true
```

**Why these settings**:
- 640x640 standard detection size
- 20% overlap to catch edge objects
- Filter tiny partial objects (15% minimum)
- Black padding for space imagery
- ~49 tiles per 4000x4000 image

---

### Example 2: Aerial Drone Footage (High Resolution)

**Goal**: Process 8K drone images for vehicle detection.

```toml
[parameters]
tile_height = 1024
tile_width = 1024
overlap_height_ratio = 0.25
overlap_width_ratio = 0.25
min_annotation_area_ratio = 0.2
min_annotation_width = 20
min_annotation_height = 20
tiling_mode = "constant"
padding_color_value = 114
```

**Why these settings**:
- Larger tiles (1024) for high-res imagery
- 25% overlap for safety
- Strict filtering (20% area, 20px minimum)
- Standard gray padding

---

### Example 3: Medical Imaging (Whole Slide Images)

**Goal**: Tile gigapixel pathology images.

```toml
[parameters]
tile_height = 512
tile_width = 512
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0
min_annotation_area_ratio = 0.0
min_annotation_width = 0
min_annotation_height = 0
tiling_mode = "drop"
fix_annotation = true
```

**Why these settings**:
- 512x512 for microscopy standards
- No overlap (tissue features are dense)
- Keep all annotations
- Drop edge tiles (slide edges are often empty)

---

### Example 4: Document Processing (Scanning)

**Goal**: Tile large scanned documents for text detection.

```toml
[parameters]
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.1
overlap_width_ratio = 0.1
min_annotation_area_ratio = 0.3
min_annotation_width = 15
min_annotation_height = 8
tiling_mode = "constant"
padding_color_value = 255
```

**Why these settings**:
- Light overlap to preserve text at boundaries
- Filter very partial text boxes (30% minimum)
- Minimum width > height (text is often wider)
- White padding for document background

---

### Example 5: Retail Shelf Monitoring

**Goal**: Tile wide shelf images for product detection.

```toml
[parameters]
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.15
overlap_width_ratio = 0.15
min_annotation_area_ratio = 0.25
min_annotation_width = 20
min_annotation_height = 20
tiling_mode = "constant"
padding_color_value = 114
```

**Why these settings**:
- Standard detection size
- Moderate overlap (products at edges)
- Filter partial products (25% minimum)
- Minimum size prevents tiny fragments

---

### Example 6: No Overlap for Speed

**Goal**: Maximum speed, accept some edge losses.

```toml
[parameters]
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.0
overlap_width_ratio = 0.0
min_annotation_area_ratio = 0.0
min_annotation_width = 0
min_annotation_height = 0
tiling_mode = "drop"
```

**Why these settings**:
- No overlap = minimal tiles
- No filtering = keep everything
- Drop mode = no padding computation
- ~4x faster than 25% overlap

---

## 🔧 Parameter Tuning Workflow

### Step 1: Calculate Tile Count

Estimate how many tiles will be created:

```
Formula (no overlap):
tiles_per_image = ceil(image_width / tile_width) × ceil(image_height / tile_height)

Example:
Image: 3840 × 2160
Tiles: 640 × 640
tiles_per_image = ceil(3840/640) × ceil(2160/640) = 6 × 4 = 24 tiles

With 20% overlap:
Effective tile step = 640 × (1 - 0.2) = 512
tiles_per_image = ceil(3840/512) × ceil(2160/512) = 8 × 5 = 40 tiles
```

### Step 2: Test on Sample Images

Run on 5-10 images first:
```toml
# Conservative starting point
tile_height = 640
tile_width = 640
overlap_height_ratio = 0.15
overlap_width_ratio = 0.15
min_annotation_area_ratio = 0.1
```

### Step 3: Review Results

Check in Picsellia:
- Are objects cut at boundaries?
- Too many partial annotations?
- Are tiny fragments cluttering the data?
- Is tile count reasonable?

### Step 4: Adjust Parameters

| Observation | Action |
|-------------|--------|
| Objects cut at edges | Increase overlap (try 0.2-0.3) |
| Too many tiles | Reduce overlap or increase tile size |
| Many tiny fragments | Increase `min_annotation_area_ratio` |
| Missing important partial objects | Decrease filtering parameters |
| Edge padding looks bad | Try different `tiling_mode` |

---

## 📊 Understanding Your Results

### Output Dataset Structure

After processing:
- **New dataset version** created
- **Tiled images** uploaded to datalake
- **Annotations** adjusted to tile coordinates
- **Original image metadata** preserved in tags/filenames

### Tile Naming Convention

Tiled images are typically named:
```
original_name_tile_X_Y.ext

Examples:
image_001_tile_0_0.jpg  (top-left tile)
image_001_tile_1_0.jpg  (second column, first row)
image_001_tile_0_1.jpg  (first column, second row)
```

### Annotation Statistics

Check pipeline logs for:
- Total tiles created
- Annotations preserved
- Annotations filtered out
- Filtering breakdown by reason

---

## ❓ Troubleshooting Guide

### Issue: Too Many Tiles Created

**Cause**: High overlap or small tile size.

**Solutions**:
1. Reduce overlap ratios (try 0.1 or 0.0)
2. Increase tile size (try 1024 instead of 640)
3. Use `tiling_mode = "drop"` to skip edge tiles

---

### Issue: Objects Cut in Half

**Cause**: Insufficient overlap.

**Solutions**:
1. Increase overlap to 0.2-0.3
2. Review object positions - may be unavoidable for very large objects
3. Consider larger tiles

---

### Issue: Many Tiny Useless Annotations

**Cause**: Filtering too lenient.

**Solutions**:
1. Increase `min_annotation_area_ratio` to 0.2-0.3
2. Set `min_annotation_width` and `min_annotation_height` (e.g., 15-20)
3. Review filtered annotations - ensure not removing important objects

---

### Issue: Important Partial Annotations Removed

**Cause**: Filtering too strict.

**Solutions**:
1. Decrease `min_annotation_area_ratio` (try 0.05-0.1)
2. Reduce or remove `min_annotation_width/height` requirements
3. Check if objects are unusually large relative to tiles

---

### Issue: Pipeline Fails on Invalid Annotations

**Check**:
1. Is `fix_annotation = true`?
2. If failures persist, some annotations may be severely malformed
3. Review original dataset quality
4. Use dataset validation tools first

---

### Issue: Edge Tiles Look Strange

**Cause**: Padding mode not suitable for your images.

**Solutions**:
```toml
# Try different modes
tiling_mode = "reflect"    # Natural boundaries
tiling_mode = "edge"       # Extend edges
tiling_mode = "drop"       # Skip edge tiles
```

---

### Issue: Memory Errors

**Cause**: Processing very high-resolution images.

**Solutions**:
1. Process dataset in smaller batches
2. Ensure sufficient system RAM
3. Consider pre-processing largest images separately

---

## 💡 Best Practices

### 1. Match Tile Size to Model

```toml
# YOLOv8 standard
tile_height = 640
tile_width = 640

# YOLOv8 small
tile_height = 416
tile_width = 416

# High-res models
tile_height = 1024
tile_width = 1024
```

### 2. Use Overlap for Detection Tasks

```toml
# Recommended for object detection
overlap_height_ratio = 0.15
overlap_width_ratio = 0.15
```

Prevents missing objects at tile boundaries.

### 3. Filter Intelligently

```toml
# Good baseline filtering
min_annotation_area_ratio = 0.15
min_annotation_width = 10
min_annotation_height = 10
```

Removes noise while preserving important objects.

### 4. Test on Samples First

Always run on 5-10 images before processing entire dataset:
- Verify tile count is reasonable
- Check annotation quality
- Confirm filtering isn't too aggressive

### 5. Preserve Original Dataset

The tiler creates a NEW dataset version. Always keep:
- Original high-resolution images
- Original annotations
- Metadata linking tiles to originals

### 6. Consider Training Efficiency

```
More overlap = Better coverage but:
- More tiles to process during training
- Longer training time
- Same objects appear multiple times

Balance overlap against training time.
```

### 7. Document Your Configuration

Record tiling parameters for reproducibility:
```
Dataset: satellite_2024_v1
Tiling: 640×640, 20% overlap
Filtering: min_area_ratio=0.15, min_size=10px
Result: 15,000 images → 180,000 tiles
```

---

## 🚀 Getting Started Checklist

- [ ] Have dataset with large images
- [ ] Decide on tile size (usually 640×640)
- [ ] Determine if overlap is needed (usually 0.15-0.2)
- [ ] Set annotation filtering thresholds
- [ ] Test on 5-10 sample images
- [ ] Review tile quality and count
- [ ] Adjust parameters if needed
- [ ] Process full dataset
- [ ] Verify output dataset
- [ ] Use tiled dataset for training

---

## 🔗 Related Pipelines

- **YOLOv8 Training**: Train models on tiled datasets
- **YOLOv8 Pre-Annotation**: Pre-annotate tiled images
- **Bounding Box Cropper**: Alternative approach for large images

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- Tiling strategy advice → Refer to this guide
- Dataset processing issues → Check troubleshooting section

---

## 🎯 Common Use Cases Summary

| Use Case | Tile Size | Overlap | Filtering |
|----------|-----------|---------|-----------|
| Satellite imagery | 640 | 0.2 | Medium (0.15) |
| Aerial photos | 1024 | 0.25 | Strict (0.2) |
| Medical images | 512 | 0.0 | None (0.0) |
| Documents | 640 | 0.1 | Moderate (0.3) |
| Retail shelves | 640 | 0.15 | Medium (0.25) |
| Fast processing | Any | 0.0 | None |

---

**Pipeline Version**: 1.0
**Type**: Dataset Version Creation
**Supported Types**: Object Detection, Segmentation, Classification
**Last Updated**: 2026-01-08

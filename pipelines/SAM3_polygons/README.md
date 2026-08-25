# SAM3_Polygons Pre-Annotation Pipeline

**Automatically segment objects in your images using natural language prompts.**

This Picsellia pipeline uses Meta's SAM-3 (Segment Anything Model 3) to generate high-quality polygon segmentation annotations without requiring any training. Simply describe what you want to segment, and SAM-3 will find and create precise polygon masks for those objects across your entire dataset.

## What You'll Get

After running this pipeline, your Picsellia dataset will contain:
- ✅ Precise polygon segmentation masks
- ✅ Properly labeled categories based on your text prompts
- ✅ COCO-format annotations ready for model training
- ✅ No training required - zero-shot segmentation

## Quick Start Guide

### 🎯 Single-Class Segmentation

**Goal**: Segment all instances of ONE object type.

```toml
[parameters]
text_prompt = "car"
threshold = 0.3
mask_threshold = 0.5
```

This will find all cars in your images and create polygon segmentation masks labeled as "car".

### 🎯 Multi-Class Segmentation

**Goal**: Segment MULTIPLE object types in one run.

```toml
[parameters]
text_prompt = "car,person,bicycle"
threshold = 0.3
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
```

This will find cars, people, and bicycles, automatically handling cases where the same object might be detected under different labels.

---

## 📋 Parameter Reference Guide

This section explains every parameter you can configure in the pipeline. Parameters are set in your Picsellia processing job configuration.

### 🔵 Essential Parameters (Required)

#### `text_prompt`
**What it does**: Tells SAM-3 what objects to segment in your images.

**Type**: Text (string)
**Default**: None
**Required**: Yes

**How to use**:
- **Single class**: `text_prompt = "car"`
- **Multiple classes**: `text_prompt = "car,person,bicycle,truck"`
  - Separate class names with commas
  - Spaces around commas are optional
  - Each class will be created as a separate category in your dataset

**Examples**:
```toml
# Waste detection
text_prompt = "plastic bottle,paper,cardboard,metal can"

# Street scene
text_prompt = "car,bus,truck,motorcycle,person,bicycle"

# Retail products
text_prompt = "can,bottle,box,package"
```

**Tips**:
- Use simple, clear descriptions (e.g., "car" not "automobile vehicle")
- Be specific when needed (e.g., "plastic bottle" vs just "bottle")
- SAM-3 understands common object names well
- More prompts = longer processing time (scales linearly)

---

#### `threshold`
**What it does**: Controls how confident SAM-3 must be before creating a segmentation mask.

**Type**: Decimal number (float)
**Range**: 0.0 to 1.0
**Default**: `0.5`
**Recommended starting point**: `0.3` to `0.4`

**How it works**:
- **Lower values (e.g., 0.2-0.3)**: More segmentations, including uncertain ones
  - ✅ Catches more objects
  - ❌ May include false positives
- **Higher values (e.g., 0.6-0.8)**: Fewer, more confident segmentations
  - ✅ Higher precision
  - ❌ May miss some valid objects

**Visual guide**:
```
threshold = 0.2  →  🔍🔍🔍🔍🔍  (Many detections, some noise)
threshold = 0.5  →  🔍🔍🔍        (Balanced)
threshold = 0.8  →  🔍            (Very few, high confidence)
```

**When to adjust**:
- **Missing objects?** → Lower the threshold (try 0.3)
- **Too many false positives?** → Raise the threshold (try 0.6)
- **Objects in cluttered scenes?** → Start lower (0.3-0.4)
- **Clean backgrounds?** → Can use higher (0.5-0.6)

**Example configurations**:
```toml
# For challenging scenes with small or partial objects
threshold = 0.25

# For clean images with clear objects
threshold = 0.6

# Balanced default
threshold = 0.4
```

---

#### `mask_threshold`
**What it does**: Controls how tight or loose the polygon segmentation masks are around objects.

**Type**: Decimal number (float)
**Range**: 0.0 to 1.0
**Default**: `0.7`
**Recommended starting point**: `0.5`

**How it works**:
- **Lower values (e.g., 0.3-0.5)**: Larger, looser polygons
  - ✅ Captures entire object including fuzzy edges
  - ❌ May include background pixels
- **Higher values (e.g., 0.6-0.8)**: Smaller, tighter polygons
  - ✅ Very precise boundaries
  - ❌ May cut off parts of the object

**Visual example**:
```
mask_threshold = 0.3  →  [████████████]  (Loose, includes edges)
mask_threshold = 0.5  →  [  ████████  ]  (Balanced)
mask_threshold = 0.8  →  [    ████    ]  (Tight, core only)
```

**When to adjust**:
- **Polygons too small?** → Lower the threshold (try 0.4-0.5)
- **Polygons include background?** → Raise the threshold (try 0.6-0.7)
- **Objects with fuzzy edges (people, animals)?** → Use lower values (0.4-0.5)
- **Objects with sharp edges (cars, boxes)?** → Can use higher values (0.6-0.7)

**Example configurations**:
```toml
# For objects with unclear boundaries (people, trees)
mask_threshold = 0.4

# For objects with sharp edges (vehicles, products)
mask_threshold = 0.6

# Very precise annotations needed
mask_threshold = 0.7
```

**Relationship with `threshold`**:
- `threshold` determines IF an object is segmented
- `mask_threshold` determines HOW MUCH of that object is included in the polygon
- They work independently - you can have high confidence detections with loose masks, or vice versa

---

### 🔵 Post-Processing Parameters

#### `min_area`
**What it does**: Filters out very small segmentations that are likely noise or irrelevant.

**Type**: Decimal number (float)
**Unit**: Square pixels
**Default**: `50.0`
**Recommended range**: `50.0` to `500.0` depending on image resolution

**How it works**:
Any segmented object with a polygon area smaller than this value will be discarded.

**When to adjust**:
- **High-resolution images (e.g., 4K)?** → Increase to 200-500
- **Low-resolution images (e.g., 640x480)?** → Keep at 50-100
- **Many tiny false positives?** → Increase significantly
- **Missing small but valid objects?** → Decrease

**Calculating the right value**:
```
Example: Segmenting cars in 1920x1080 images
- A small car might be 100x80 pixels = 8,000 px²
- Set min_area = 1000 to filter noise while keeping cars
```

**Example configurations**:
```toml
# High-res images, filtering small noise
min_area = 300.0

# Low-res images, keep small objects
min_area = 50.0

# Segment only large objects (e.g., vehicles in aerial imagery)
min_area = 1000.0
```

---

#### `max_overlap_ratio`
**What it does**: Removes duplicate segmentations OF THE SAME CLASS that overlap too much.

**Type**: Decimal number (float)
**Range**: 0.0 to 1.0
**Default**: `0.3`

**How it works**:
If two segmentations of the SAME class (e.g., two "car" detections) overlap more than this ratio, the smaller one is removed.

**When to adjust**:
- **SAM-3 segmenting the same object multiple times?** → Lower to 0.2
- **Valid adjacent objects being removed?** → Raise to 0.4-0.5
- Usually the default (0.3) works well

**Note**: This only affects segmentations within the same class. For cross-class overlaps (e.g., "car" vs "vehicle"), see the Multi-Class Deduplication parameters below.

---

### 🔵 Multi-Class Deduplication Parameters

**⚠️ Only relevant when using multiple text prompts** (e.g., `text_prompt = "car,person,bicycle"`).

When SAM-3 runs inference for multiple classes, it might segment the same object multiple times with different labels. For example, a person standing in front of a car might be segmented as both "person" (small mask) and "car" (large mask including the person).

These parameters control how the pipeline handles such conflicts.

---

#### `iou_threshold`
**What it does**: Detects overlapping segmentations of DIFFERENT classes that are roughly the same size.

**Type**: Decimal number (float)
**Range**: 0.0 to 1.0
**Default**: `0.5`
**Recommended range**: `0.4` to `0.7`

**How it works**:
- Calculates IoU (Intersection over Union) between all pairs of segmentations
- IoU = (overlap area) / (total area covered by both polygons)
- If IoU > this threshold, marks them as duplicates

**Visual understanding**:
```
Two similar-sized masks with high overlap:

Mask A (person): [  ████████  ]
Mask B (car):    [   ███████   ]
                    ↑ high overlap

IoU = 0.6 → If threshold = 0.5, considered duplicates
```

**When to adjust**:
- **Too many valid segmentations removed?** → Lower (try 0.3-0.4)
- **Same object segmented with multiple labels?** → Raise (try 0.6-0.7)
- **Densely packed objects (crowd scenes)?** → Lower (try 0.3)
- **Sparse scenes?** → Can use higher (try 0.6)

**Example configurations**:
```toml
# Crowded scenes, preserve more segmentations
iou_threshold = 0.3

# Clean scenes, aggressive deduplication
iou_threshold = 0.7

# Balanced default
iou_threshold = 0.5
```

---

#### `containment_threshold`
**What it does**: Detects when one segmentation is nested inside another (different sizes).

**Type**: Decimal number (float)
**Range**: 0.0 to 1.0
**Default**: `0.8`
**Recommended range**: `0.6` to `0.9`

**How it works**:
- Checks if a smaller polygon is contained within a larger one
- Containment = (overlap area) / (smaller polygon area)
- If containment > this threshold, marks them as duplicates

**Visual understanding**:
```
Small mask inside large mask:

Large mask (car):    [████████████████]
Small mask (person):      [███]
                           ↑ nested inside

Containment = 0.85 → If threshold = 0.8, considered duplicates
```

**When to adjust**:
- **Legitimate small objects inside large ones being removed?** → Raise (try 0.9)
  - Example: Person inside car (passenger) - if you want both, raise threshold
- **Same object segmented at different scales?** → Lower (try 0.6-0.7)
- **Objects in front of background objects?** → Adjust based on desired behavior

**Example configurations**:
```toml
# Keep objects even when they overlap significantly
containment_threshold = 0.95

# Aggressive removal of nested duplicates
containment_threshold = 0.7

# Balanced default
containment_threshold = 0.8
```

**Real-world scenarios**:

**Scenario 1: Person in front of car**
```
Car mask (large): Includes the car + person in front
Person mask (small): Just the person

If containment > threshold → Keep one based on strategy
```
- Want the person? → Use `deduplication_strategy = "keep_smaller"`
- Want the whole scene? → Use `deduplication_strategy = "keep_larger"`

**Scenario 2: Logo on a box**
```
Box mask (large): The entire box
Logo mask (small): Just the logo region

If containment > threshold → Keep one
```
- If segmenting products, probably want the box (keep_larger)
- If segmenting logos, want the logo (keep_smaller)

---

#### `deduplication_strategy`
**What it does**: Decides which segmentation to keep when duplicates are found.

**Type**: Text (string)
**Options**: `"keep_smaller"` or `"keep_larger"`
**Default**: `"keep_smaller"`

**How it works**:
When the pipeline identifies duplicate segmentations (via `iou_threshold` or `containment_threshold`), this parameter determines which one survives.

---

**Option 1: `"keep_smaller"` (Recommended Default)**

**Best for**:
- Segmenting foreground objects in front of backgrounds
- Scenarios where SAM-3 over-segments (includes too much)
- Precise object boundaries are important

**Behavior**: Prioritizes smaller, tighter polygons over larger ones.

**Example scenario**:
```
Scene: Person standing in front of a car

Segmentations:
- "person": Small polygon around just the person
- "car": Large polygon including car + person

Result with keep_smaller:
✅ KEEP: "person" polygon (smaller, more precise)
❌ REMOVE: "car" polygon (larger, includes both)
```

**Use cases**:
- Street scenes with pedestrians and vehicles
- Products on shelves (individual items vs shelf)
- Animals in front of landscape elements
- Any scenario where you want individual objects, not grouped regions

---

**Option 2: `"keep_larger"`**

**Best for**:
- Full object coverage is more important than precision
- Smaller segmentations are often fragments or false positives
- Detecting complete regions/groups

**Behavior**: Prioritizes larger, more complete polygons over smaller ones.

**Example scenario**:
```
Scene: Person standing in front of a car

Segmentations:
- "person": Small polygon around just the person
- "car": Large polygon including car + person

Result with keep_larger:
❌ REMOVE: "person" polygon (smaller)
✅ KEEP: "car" polygon (larger, complete scene element)
```

**Use cases**:
- Warehouse inventory (full pallet vs individual boxes)
- Aerial/satellite imagery (complete buildings vs partial walls)
- Medical imaging (complete organs vs small tissue regions)
- When small segmentations are usually noise

---

**Decision Matrix**:

| Your Priority | Recommended Strategy |
|---------------|---------------------|
| Segment individual objects in complex scenes | `keep_smaller` |
| Segment foreground elements | `keep_smaller` |
| Maximize precision over coverage | `keep_smaller` |
| Segment complete objects/regions | `keep_larger` |
| Small segmentations are usually false positives | `keep_larger` |
| Maximize coverage over precision | `keep_larger` |

**Example configurations**:
```toml
# Retail: Segment individual products on a shelf
deduplication_strategy = "keep_smaller"

# Warehouse: Segment full pallets, not individual boxes
deduplication_strategy = "keep_larger"

# Street scenes: Segment people and vehicles separately
deduplication_strategy = "keep_smaller"
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Autonomous Driving Dataset

**Goal**: Segment traffic scenes with vehicles, pedestrians, and infrastructure.

```toml
[parameters]
text_prompt = "car,truck,bus,motorcycle,person,bicycle,traffic light,stop sign"
threshold = 0.35
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
min_area = 100.0
```

**Why these settings**:
- `threshold = 0.35`: Slightly sensitive to catch partially occluded objects
- `mask_threshold = 0.5`: Balanced polygons for both vehicles (sharp edges) and people (fuzzy boundaries)
- `deduplication_strategy = "keep_smaller"`: Separate pedestrians from vehicles they're in front of
- `min_area = 100.0`: Filter tiny noise while keeping distant small objects

---

### Example 2: Warehouse Inventory

**Goal**: Segment full pallets and large equipment, ignore individual boxes.

```toml
[parameters]
text_prompt = "wooden pallet,cardboard box,forklift"
threshold = 0.4
mask_threshold = 0.6
iou_threshold = 0.6
containment_threshold = 0.75
deduplication_strategy = "keep_larger"
min_area = 500.0
```

**Why these settings**:
- `threshold = 0.4`: Higher confidence for cleaner warehouse environment
- `mask_threshold = 0.6`: Tight polygons for industrial objects with clear edges
- `deduplication_strategy = "keep_larger"`: Prioritize full pallets over individual boxes
- `min_area = 500.0`: Filter out small objects (individual items, debris)
- `iou_threshold = 0.6`: More aggressive deduplication in organized spaces

---

### Example 3: Recycling Bin Monitoring

**Goal**: Segment different types of recyclable waste items.

```toml
[parameters]
text_prompt = "plastic bottle,glass bottle,aluminum can,paper,cardboard"
threshold = 0.3
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
min_area = 50.0
max_overlap_ratio = 0.25
```

**Why these settings**:
- `threshold = 0.3`: Lower to catch partially visible items in bin
- `mask_threshold = 0.5`: Balanced for various material types
- `deduplication_strategy = "keep_smaller"`: Separate overlapping items
- `min_area = 50.0`: Keep even small items (bottle caps, crushed cans)
- `max_overlap_ratio = 0.25`: Stricter same-class deduplication for cluttered bins

---

### Example 4: Retail Shelf Monitoring

**Goal**: Segment and count individual products on shelves with high precision.

```toml
[parameters]
text_prompt = "can,bottle,box,package"
threshold = 0.4
mask_threshold = 0.6
iou_threshold = 0.4
containment_threshold = 0.75
deduplication_strategy = "keep_smaller"
min_area = 75.0
max_overlap_ratio = 0.2
```

**Why these settings**:
- `threshold = 0.4`: Balanced for varied product types
- `mask_threshold = 0.6`: Tight polygons for products with clear packaging
- `iou_threshold = 0.4`: Lower to preserve adjacent similar products
- `deduplication_strategy = "keep_smaller"`: Count individual items, not groups
- `max_overlap_ratio = 0.2`: Strict to prevent counting same product twice

---

### Example 5: Medical Imaging (Generic Objects)

**Goal**: Segment specific anatomical structures or devices.

```toml
[parameters]
text_prompt = "catheter,surgical tool"
threshold = 0.5
mask_threshold = 0.7
iou_threshold = 0.7
containment_threshold = 0.85
deduplication_strategy = "keep_smaller"
min_area = 30.0
```

**Why these settings**:
- `threshold = 0.5`: Higher confidence for medical accuracy
- `mask_threshold = 0.7`: Very precise polygons for clinical use
- High deduplication thresholds: Conservative to avoid false merging
- `min_area = 30.0`: Lower to detect small devices

---

### Example 6: Aerial/Satellite Imagery

**Goal**: Segment buildings, vehicles, and infrastructure from above.

```toml
[parameters]
text_prompt = "building,car,parking lot,road"
threshold = 0.45
mask_threshold = 0.6
iou_threshold = 0.6
containment_threshold = 0.8
deduplication_strategy = "keep_larger"
min_area = 200.0
```

**Why these settings**:
- `threshold = 0.45`: Moderate for varied aerial perspectives
- `deduplication_strategy = "keep_larger"`: Prefer complete structures over partial detections
- `min_area = 200.0`: Filter noise from high-resolution satellite imagery
- `iou_threshold = 0.6`: Higher due to clear separation in aerial view

---

### Example 7: Instance Segmentation for Training Data

**Goal**: Create high-quality polygon annotations for training segmentation models.

```toml
[parameters]
text_prompt = "object"
threshold = 0.4
mask_threshold = 0.5
min_area = 100.0
```

**Why these settings**:
- Single class `"object"` for generic instance segmentation
- Balanced parameters for high-quality training data
- Can be post-processed or relabeled in Picsellia

---

## 🔧 Parameter Tuning Workflow

**Start with these defaults** for most use cases:
```toml
[parameters]
text_prompt = "your,classes,here"
threshold = 0.35
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
min_area = 100.0
```

**Then iterate**:

1. **Run on a small sample** (10-50 images)
2. **Review results in Picsellia**
3. **Adjust based on issues**:

   | Problem | Solution |
   |---------|----------|
   | Missing many objects | Lower `threshold` (try 0.25-0.3) |
   | Too many false positives | Raise `threshold` (try 0.5-0.6) |
   | Polygons too small | Lower `mask_threshold` (try 0.4) |
   | Polygons include background | Raise `mask_threshold` (try 0.6-0.7) |
   | Same object, multiple labels | Adjust `iou_threshold` and `containment_threshold` |
   | Wrong labels on objects | Try opposite `deduplication_strategy` |
   | Tiny noise segmentations | Raise `min_area` |

4. **Re-run with adjusted parameters**
5. **Repeat until satisfied**
6. **Scale to full dataset**

---

## 📊 Understanding Your Results

After the pipeline completes, check your Picsellia dataset:

### Expected Output

- **Categories**: One per text prompt (e.g., "car", "person", "bicycle")
- **Annotations per image**: Varies based on image content and parameters
- **Format**: COCO with polygon segmentation masks
- **Quality indicators**:
  - Check a few images manually
  - Look for missed objects → adjust `threshold`
  - Look for false positives → adjust `threshold` or `min_area`
  - Check label accuracy → adjust deduplication parameters
  - Check polygon quality → adjust `mask_threshold`

### Performance Notes

- **Processing time**: ~2-5 seconds per image per text prompt (GPU)
- **Multi-class scaling**: 5 prompts = ~5x processing time
- **GPU recommended**: CPU processing is 10-20x slower
- **Memory usage**: ~4-8 GB VRAM for standard images

---

## ❓ Troubleshooting Guide

### Issue: Missing Objects

**Check**:
1. Is `threshold` too high? Try lowering to 0.3
2. Are objects smaller than `min_area`? Lower it
3. Is your text prompt clear? Try alternative descriptions

### Issue: False Positives

**Check**:
1. Is `threshold` too low? Try raising to 0.5-0.6
2. Are false positives very small? Raise `min_area`
3. Is `mask_threshold` too low? Try raising to 0.6

### Issue: Same Object Segmented Multiple Times

**Check**:
1. Using multi-class? Adjust `iou_threshold` and `containment_threshold`
2. Same class duplicates? Check `max_overlap_ratio`
3. Try switching `deduplication_strategy`

### Issue: Wrong Labels on Objects

**Check**:
1. Is `deduplication_strategy` correct for your use case?
2. Adjust `containment_threshold` (higher = less aggressive deduplication)
3. Review text prompts for ambiguity

### Issue: Polygons Too Tight or Too Loose

**Solution**: Adjust `mask_threshold`
- Too tight? Lower to 0.4-0.5
- Too loose? Raise to 0.6-0.7

### Issue: Slow Processing

**Check**:
1. Using GPU? Processing is much faster
2. Too many text prompts? Consider splitting into multiple jobs
3. Raise `threshold` to reduce segmentations per image
4. Raise `min_area` to filter earlier in pipeline

### Issue: Pipeline Fails with Dataset Type Error

**Check**:
1. Your dataset type must be either `NOT_CONFIGURED` or `SEGMENTATION`
2. Other dataset types (e.g., `OBJECT_DETECTION`, `CLASSIFICATION`) are not supported
3. Create a new dataset version or change the dataset type before running the pipeline

---

## 🚀 Getting Started Checklist

- [ ] Define your object classes (text prompts)
- [ ] Ensure your dataset type is `NOT_CONFIGURED` or `SEGMENTATION`
- [ ] Start with recommended defaults
- [ ] Run on 10-20 sample images
- [ ] Review polygon quality in Picsellia
- [ ] Adjust parameters based on findings
- [ ] Run on full dataset
- [ ] Export annotations for training

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- Pipeline configuration help → Refer to this guide
- SAM-3 model questions → See [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3)

---

**Pipeline Version**: 1.0.0
**Type**: Pre-annotation (Segmentation)
**Last Updated**: 2026-01-07

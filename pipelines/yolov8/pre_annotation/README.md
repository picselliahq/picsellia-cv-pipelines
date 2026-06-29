# YOLOv8 Pre-Annotation Pipeline

**Automatically annotate your images using a trained YOLOv8 model.**

This Picsellia pipeline uses a trained YOLOv8 object detection model to generate bounding box annotations on unlabeled images. Perfect for bootstrapping new datasets, augmenting existing annotations, or accelerating your labeling workflow.

## What You'll Get

After running this pipeline, your Picsellia dataset will contain:
- ✅ Bounding box annotations for detected objects
- ✅ Confidence scores for each detection
- ✅ Labels matching your model's categories
- ✅ COCO-format annotations ready for review or training

---

## Inputs vs Parameters

| Mechanism | Set when | Examples |
|-----------|----------|---------|
| **Inputs** | Launching the job on Picsellia | model version, weights file name, confidence threshold |
| **Parameters** | Configuring the processing | batch size, image size, label strategy, NMS |

---

## 📥 Inputs Reference

Inputs are configured when launching the processing job from the Picsellia platform.

### `model_version`
**What it does**: The YOLOv8 model version to use for inference.

**Type**: Model Version
**Required**: Yes

Select the trained model version from your Picsellia workspace.

---

### `model_file_name`
**What it does**: Name of the weights artifact within the selected model version.

**Type**: Text
**Required**: Yes

**Example**: `best-model`, `pretrained-weights`, `last`

**Note**: The file must be in PyTorch format (`.pt`). ONNX models are not supported for pre-annotation.

---

### `confidence_threshold`
**What it does**: Minimum confidence score for a detection to be saved as an annotation.

**Type**: Number
**Required**: Yes
**Recommended range**: `0.2` to `0.5`

**How it works**:
- **Lower values (0.1-0.2)**: More detections, including uncertain ones — catch more objects but more false positives to review
- **Higher values (0.4-0.6)**: Fewer, high-confidence detections — more accurate but may miss some valid objects

**When to adjust**:
- Starting a new dataset → `0.2-0.3` to catch more objects
- Augmenting existing labels → `0.4-0.5` to avoid duplicates
- High-quality model → can use lower threshold

---

## 📋 Parameters Reference

### `batch_size`
**What it does**: Number of images processed simultaneously during inference.

**Type**: Integer
**Default**: `8`

**GPU Memory Guide**:
```
4 GB VRAM  → batch_size = 4
8 GB VRAM  → batch_size = 8-16
12 GB VRAM → batch_size = 16-32
24 GB VRAM → batch_size = 32-64
```

---

### `image_size`
**What it does**: Size to which images are resized during inference.

**Type**: Integer
**Default**: `640`
**Common values**: `320`, `416`, `640`, `1280`

**Important**: Should match the size used during model training for best results.

- **Smaller (320-416)**: Faster inference, may miss small objects
- **Medium (640)**: Balanced — most common
- **Larger (1280)**: Better small object detection, slower inference

---

### `device`
**What it does**: Hardware device for running inference.

**Type**: String
**Default**: `"cuda"`
**Options**: `"cuda:0"`, `"cuda:1"`, `"cpu"`

```toml
# Use first GPU (recommended)
device = "cuda:0"

# Use CPU (very slow, not recommended)
device = "cpu"
```

---

### `label_matching_strategy`
**What it does**: How to handle labels that don't exist in the target dataset.

**Type**: String
**Default**: `"add"`
**Options**: `"add"`, `"skip"`, `"strict"`

| Option | Behaviour | Best For |
|--------|-----------|----------|
| `"add"` | Creates new labels automatically | New datasets, bootstrapping |
| `"skip"` | Ignores detections with unknown labels | Controlled label vocabulary |
| `"strict"` | Fails if any label mismatch | Production pipelines, QA |

**Example scenario**:
```
Model detects: "car", "person", "bicycle"
Dataset has:   "car", "person"

"add"    → car ✅  person ✅  bicycle ✅ (label created)
"skip"   → car ✅  person ✅  bicycle ❌ (skipped)
"strict" → ❌ Pipeline fails
```

---

### `replace_annotations`
**What it does**: Whether to delete existing annotations before adding new ones.

**Type**: Boolean
**Default**: `false`

**⚠️ Warning**: Setting this to `true` will permanently delete existing annotations. Always create a new dataset version as a backup first.

---

### `agnostic_nms`
**What it does**: Enable class-agnostic Non-Maximum Suppression.

**Type**: Boolean
**Default**: `true`

- **`true`**: NMS considers all classes together — removes overlapping boxes regardless of class, prevents duplicate detections
- **`false`**: NMS works per-class — allows overlapping boxes from different classes

---

## Quick Start Guide

### 🎯 Basic Pre-Annotation

**Inputs** (set when launching the job):

| Input | Value |
|-------|-------|
| `model_version` | *(select your trained model)* |
| `model_file_name` | `best-model` |
| `confidence_threshold` | `0.25` |

**Parameters**:
```toml
[parameters]
batch_size = 8
image_size = 640
device = "cuda:0"
label_matching_strategy = "add"
replace_annotations = false
agnostic_nms = true
```

---

### 🎯 High-Confidence Annotations Only

**Inputs**: `confidence_threshold = 0.5`

**Parameters**:
```toml
[parameters]
batch_size = 8
image_size = 640
device = "cuda:0"
agnostic_nms = true
```

Higher threshold means fewer but more accurate annotations.

---

### 🎯 Replace Existing Annotations

**Inputs**: `confidence_threshold = 0.25`, `model_file_name = best-model`

**Parameters**:
```toml
[parameters]
replace_annotations = true
label_matching_strategy = "add"
```

⚠️ Creates a new dataset version backup before running.

---

## 🎓 Real-World Configuration Examples

### Example 1: Bootstrapping New Dataset

**Goal**: Quickly annotate thousands of unlabeled images.

**Inputs**: `confidence_threshold = 0.2`, `model_file_name = pretrained-weights`

**Parameters**:
```toml
[parameters]
batch_size = 16
image_size = 640
device = "cuda:0"
label_matching_strategy = "add"
replace_annotations = false
agnostic_nms = true
```

Lower confidence to catch more objects, higher batch size for speed.

---

### Example 2: Augmenting Partially Labeled Dataset

**Goal**: Add annotations to previously unlabeled images.

**Inputs**: `confidence_threshold = 0.35`, `model_file_name = best-model`

**Parameters**:
```toml
[parameters]
batch_size = 8
image_size = 640
device = "cuda:0"
label_matching_strategy = "skip"
replace_annotations = false
agnostic_nms = true
```

`"skip"` ensures only existing label classes get annotated.

---

### Example 3: High-Quality Pre-Annotation

**Goal**: Generate high-precision annotations that need minimal review.

**Inputs**: `confidence_threshold = 0.5`, `model_file_name = production-model`

**Parameters**:
```toml
[parameters]
batch_size = 8
image_size = 1280
device = "cuda:0"
label_matching_strategy = "add"
replace_annotations = false
agnostic_nms = true
```

High confidence + larger image size for better precision.

---

### Example 4: Fast Processing for Large Dataset

**Goal**: Process 100,000+ images as quickly as possible.

**Inputs**: `confidence_threshold = 0.3`, `model_file_name = pretrained-weights`

**Parameters**:
```toml
[parameters]
batch_size = 32
image_size = 416
device = "cuda:0"
label_matching_strategy = "add"
agnostic_nms = true
```

Large batch + smaller image size = ~5-10 images/second on modern GPU.

---

### Example 5: Specific Class Pre-Annotation

**Goal**: Only annotate classes that already exist in the dataset.

**Inputs**: `confidence_threshold = 0.3`, `model_file_name = multi-class-model`

**Parameters**:
```toml
[parameters]
batch_size = 8
image_size = 640
device = "cuda:0"
label_matching_strategy = "skip"
```

```
Model can detect: car, truck, bus, person, bicycle
Dataset only has: car, truck, bus
Result: Only car, truck, bus are annotated
```

---

## 🔧 Tuning Workflow

### Step 1: Test on Small Sample

Run on 10-20 images first with a moderate confidence threshold.

**Inputs**: `confidence_threshold = 0.25`

**Parameters**:
```toml
batch_size = 4
image_size = 640
```

### Step 2: Adjust Based on Results

| Observation | Action |
|-------------|--------|
| Missing many objects | Lower `confidence_threshold` input (try 0.15-0.2) |
| Too many false positives | Raise `confidence_threshold` input (try 0.4-0.5) |
| Wrong labels appearing | Adjust `label_matching_strategy` parameter |
| Duplicate detections | Ensure `agnostic_nms = true` |
| Processing too slow | Increase `batch_size` or decrease `image_size` |
| Out of GPU memory | Decrease `batch_size` |

---

## ❓ Troubleshooting Guide

### No annotations created
1. Is `confidence_threshold` input too high? Try `0.1`
2. Does the model match the dataset task?
3. Check logs for errors

### Too many false positives
1. Raise `confidence_threshold` input to `0.4`-`0.5`
2. Verify model quality

### Missing many objects
1. Lower `confidence_threshold` input to `0.15`-`0.2`
2. Increase `image_size` to `1280` for small objects

### Duplicate/overlapping boxes
Set `agnostic_nms = true` in parameters.

### Pipeline fails — "Cannot use ONNX model"
The `model_file_name` input must point to a `.pt` file, not `.onnx`.

### Out of memory
Reduce `batch_size` (try `4` or `2`) or reduce `image_size` to `416`.

---

## 💡 Best Practices

1. **Always test first** — run on 10-20 images before the full dataset
2. **Tune `confidence_threshold` as an input** — sweet spot is `0.2`-`0.35` for most cases
3. **Backup before replace** — create a new dataset version before setting `replace_annotations = true`
4. **Match training image size** — set `image_size` to what your model was trained with
5. **Use GPU** — CPU inference is 10-20× slower: `device = "cuda:0"`
6. **Iterate** — use corrected annotations to train a better model, then pre-annotate again

---

## 🚀 Getting Started Checklist

- [ ] Have a trained YOLOv8 model version in Picsellia
- [ ] Model weights are in `.pt` format (not ONNX)
- [ ] Select the model version input when launching the job
- [ ] Set `model_file_name` input to the correct artifact name
- [ ] Set `confidence_threshold` input (start with `0.25`)
- [ ] Configure parameters (batch size, image size, label strategy)
- [ ] Test on 10-20 images first
- [ ] Review results and adjust `confidence_threshold` if needed
- [ ] Run on full dataset
- [ ] Review and correct annotations in Picsellia

---

## 🔄 Integration with Training Pipeline

```
1. Pre-annotate with base model (this pipeline)
   ↓
2. Review and correct annotations
   ↓
3. Train new model (YOLOv8 Training Pipeline)
   ↓
4. Pre-annotate new images with improved model
   ↓
5. Repeat — each cycle improves quality
```

---

**Pipeline Version**: 1.0.11
**Type**: Pre-Annotation
**Framework**: Ultralytics YOLOv8

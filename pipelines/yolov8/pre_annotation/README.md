# YOLOv8 Pre-Annotation Pipeline

**Automatically annotate your images using a trained YOLOv8 model.**

This Picsellia pipeline uses a trained YOLOv8 object detection model to generate bounding box annotations on unlabeled images. Perfect for bootstrapping new datasets, augmenting existing annotations, or accelerating your labeling workflow.

## What You'll Get

After running this pipeline, your Picsellia dataset will contain:
- ✅ Bounding box annotations for detected objects
- ✅ Confidence scores for each detection
- ✅ Labels matching your model's categories
- ✅ COCO-format annotations ready for review or training

## Quick Start Guide

### 🎯 Basic Pre-Annotation

**Goal**: Annotate images with a trained YOLOv8 model using default settings.

```toml
[parameters]
model_file_name = "pretrained-weights"
confidence_threshold = 0.25
batch_size = 8
image_size = 640
device = "cuda:0"
```

This will detect objects with at least 25% confidence and add annotations to your dataset.

### 🎯 High-Confidence Annotations Only

**Goal**: Only keep very confident predictions to minimize false positives.

```toml
[parameters]
model_file_name = "pretrained-weights"
confidence_threshold = 0.5
batch_size = 8
image_size = 640
device = "cuda:0"
```

Higher threshold means fewer but more accurate annotations.

### 🎯 Replace Existing Annotations

**Goal**: Overwrite existing annotations with new predictions.

```toml
[parameters]
model_file_name = "pretrained-weights"
confidence_threshold = 0.25
replace_annotations = true
label_matching_strategy = "add"
```

Use with caution - this will delete existing annotations on the images.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Parameters

#### `model_file_name`
**What it does**: Name of the model weights artifact to use for predictions.

**Type**: String
**Default**: `"pretrained-weights"`

**How to use**: This should match the artifact name of your trained model weights in Picsellia.

**Example**:
```toml
# Use default pretrained weights
model_file_name = "pretrained-weights"

# Use custom artifact name
model_file_name = "best-model"
```

**Note**: The model must be in PyTorch format (.pt). ONNX models are not supported for pre-annotation.

---

#### `confidence_threshold`
**What it does**: Minimum confidence score for a detection to be saved as an annotation.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.1`
**Recommended range**: `0.2` to `0.5`

**How it works**:
- **Lower values (0.1-0.2)**: More detections, including uncertain ones
  - ✅ Catch more objects
  - ❌ More false positives to review
  
- **Higher values (0.4-0.6)**: Fewer, high-confidence detections
  - ✅ More accurate annotations
  - ❌ May miss some valid objects

**Visual guide**:
```
threshold = 0.1  →  📦📦📦📦📦  (Many detections, some noise)
threshold = 0.25 →  📦📦📦      (Balanced - recommended)
threshold = 0.5  →  📦          (High confidence only)
```

**When to adjust**:
- **Starting a new dataset?** → Use 0.2-0.3 to catch more objects
- **Augmenting existing labels?** → Use 0.4-0.5 to avoid duplicates
- **High-quality model?** → Can use lower threshold
- **Model still learning?** → Use higher threshold

**Example configurations**:
```toml
# Maximize recall - get all possible detections
confidence_threshold = 0.15

# Balanced approach (recommended)
confidence_threshold = 0.25

# High precision - only very confident predictions
confidence_threshold = 0.5
```

---

#### `batch_size`
**What it does**: Number of images processed simultaneously during inference.

**Type**: Integer
**Default**: `8`
**Recommended range**: `4` to `32`

**How it affects processing**:
- **Larger batch (16-32)**: 
  - ✅ Faster processing
  - ❌ Requires more GPU memory
  
- **Smaller batch (4-8)**: 
  - ✅ Works with limited GPU memory
  - ❌ Slower processing

**GPU Memory Guide**:
```
4 GB VRAM  → batch_size = 4
8 GB VRAM  → batch_size = 8-16
12 GB VRAM → batch_size = 16-32
24 GB VRAM → batch_size = 32-64
```

**Example**:
```toml
batch_size = 8
```

---

#### `image_size`
**What it does**: Size to which images are resized during inference.

**Type**: Integer
**Default**: `640`
**Common values**: `320`, `416`, `640`, `1280`

**Important**: Should match the size used during model training for best results.

**Trade-offs**:
- **Smaller (320-416)**: Faster inference, may miss small objects
- **Medium (640)**: Balanced (most common)
- **Larger (1280)**: Better small object detection, slower inference

**Example**:
```toml
# Standard inference
image_size = 640

# Fast inference for large datasets
image_size = 416

# High-quality detection for small objects
image_size = 1280
```

---

#### `device`
**What it does**: Hardware device for running inference.

**Type**: String
**Default**: `"cuda"`
**Options**: `"cuda:0"`, `"cuda:1"`, `"cpu"`

**Example**:
```toml
# Use first GPU (recommended)
device = "cuda:0"

# Use second GPU
device = "cuda:1"

# Use CPU (very slow, not recommended)
device = "cpu"
```

---

### 🔵 Label Management Parameters

#### `label_matching_strategy`
**What it does**: How to handle labels that don't exist in the target dataset.

**Type**: String
**Default**: `"add"`
**Options**: `"add"`, `"skip"`, `"strict"`

**How each option works**:

**Option 1: `"add"` (Recommended Default)**

**Behavior**: Automatically creates new labels in the dataset if they don't exist.

**Best for**:
- New unlabeled datasets
- Bootstrapping annotation projects
- When you trust the model's label set

**Example scenario**:
```
Model detects: "car", "person", "bicycle"
Dataset has: "car", "person"

Result with "add":
✅ "car" → Added (label exists)
✅ "person" → Added (label exists)
✅ "bicycle" → NEW label created, annotation added
```

---

**Option 2: `"skip"`**

**Behavior**: Ignores detections whose labels don't exist in the dataset.

**Best for**:
- Adding annotations for specific classes only
- When dataset has a controlled label set
- Filtering model output

**Example scenario**:
```
Model detects: "car", "person", "bicycle"
Dataset has: "car", "person"

Result with "skip":
✅ "car" → Added (label exists)
✅ "person" → Added (label exists)
❌ "bicycle" → Skipped (label doesn't exist)
```

---

**Option 3: `"strict"`**

**Behavior**: Requires exact match between model labels and dataset labels. Fails if any mismatch.

**Best for**:
- Production pipelines with strict requirements
- Quality control
- Validation that model matches dataset

**Example scenario**:
```
Model detects: "car", "person", "bicycle"
Dataset has: "car", "person"

Result with "strict":
❌ Pipeline fails - "bicycle" label doesn't exist in dataset
```

---

**Decision Matrix**:

| Your Situation | Recommended Strategy |
|----------------|---------------------|
| New empty dataset | `"add"` |
| Adding to partially labeled dataset | `"add"` |
| Only want specific classes | `"skip"` |
| Strict label control required | `"strict"` |
| Trust the model completely | `"add"` |
| Dataset schema is locked | `"skip"` or `"strict"` |

**Example configurations**:
```toml
# Flexible - create labels as needed
label_matching_strategy = "add"

# Conservative - only annotate existing label classes
label_matching_strategy = "skip"

# Strict - ensure perfect label alignment
label_matching_strategy = "strict"
```

---

#### `replace_annotations`
**What it does**: Whether to delete existing annotations before adding new ones.

**Type**: Boolean
**Default**: `false`

**⚠️ Warning**: Setting this to `true` will permanently delete existing annotations!

**When to use**:

**Use `false` (Default)**:
- Adding annotations to unlabeled images
- Augmenting existing human annotations
- Iterative annotation workflow
- Preserving manual work

**Use `true`**:
- Completely re-annotating with new model
- Fixing incorrect bulk annotations
- Starting fresh with better model
- You have backups of original annotations

**Example**:
```toml
# Safe - keep existing annotations
replace_annotations = false

# Dangerous - delete all existing annotations first
replace_annotations = true
```

**Best Practice**: Always create a new dataset version before using `replace_annotations = true`.

---

#### `agnostic_nms`
**What it does**: Enable class-agnostic Non-Maximum Suppression.

**Type**: Boolean
**Default**: `true`

**How it works**:

**`agnostic_nms = true` (Recommended)**:
- NMS considers all classes together
- Removes overlapping boxes regardless of class
- Better for densely packed objects
- Prevents duplicate detections of same object with different labels

**Example**: Car detected as both "car" and "vehicle" - only keeps one

**`agnostic_nms = false`**:
- NMS works per-class
- Allows overlapping boxes from different classes
- Better when objects of different classes truly overlap

**Example**: Person in front of car - keeps both boxes

**When to adjust**:
- **Default (true)** works for most cases
- Use **false** if you need to detect overlapping objects of different classes
- Use **true** to reduce duplicate/redundant annotations

**Example**:
```toml
# Recommended - prevents duplicates
agnostic_nms = true

# Allow overlapping detections from different classes
agnostic_nms = false
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Bootstrapping New Dataset

**Goal**: Quickly annotate thousands of unlabeled images to start labeling workflow.

```toml
[parameters]
model_file_name = "pretrained-weights"
confidence_threshold = 0.2
batch_size = 16
image_size = 640
device = "cuda:0"
label_matching_strategy = "add"
replace_annotations = false
agnostic_nms = true
```

**Why these settings**:
- Lower confidence threshold to catch more objects
- Higher batch size for speed
- "add" strategy to create all necessary labels
- Keep existing annotations (safe default)

**Workflow**:
1. Run pre-annotation
2. Review and correct annotations in Picsellia
3. Export for training

---

### Example 2: Augmenting Partially Labeled Dataset

**Goal**: Add annotations to images that were previously unlabeled.

```toml
[parameters]
model_file_name = "best-model"
confidence_threshold = 0.35
batch_size = 8
image_size = 640
device = "cuda:0"
label_matching_strategy = "skip"
replace_annotations = false
agnostic_nms = true
```

**Why these settings**:
- Medium confidence to avoid conflicts with human labels
- "skip" to only annotate existing label classes
- Preserve existing manual annotations
- Conservative batch size

**Workflow**:
1. Filter to unlabeled images only
2. Run pre-annotation
3. Review automatic annotations
4. Blend with existing manual labels

---

### Example 3: High-Quality Pre-Annotation

**Goal**: Generate high-precision annotations that need minimal review.

```toml
[parameters]
model_file_name = "production-model"
confidence_threshold = 0.5
batch_size = 8
image_size = 1280
device = "cuda:0"
label_matching_strategy = "add"
replace_annotations = false
agnostic_nms = true
```

**Why these settings**:
- High confidence threshold (0.5) for precision
- Larger image size for better detection
- Only very confident predictions get added

**Use case**:
- Production-grade model available
- Quality over quantity
- Minimize false positives

---

### Example 4: Fast Processing for Large Dataset

**Goal**: Process 100,000+ images as quickly as possible.

```toml
[parameters]
model_file_name = "pretrained-weights"
confidence_threshold = 0.3
batch_size = 32
image_size = 416
device = "cuda:0"
label_matching_strategy = "add"
replace_annotations = false
agnostic_nms = true
```

**Why these settings**:
- Large batch size for throughput
- Smaller image size for speed
- Balanced confidence threshold

**Performance**:
- ~5-10 images/second on modern GPU
- 100K images in 3-6 hours

---

### Example 5: Re-Annotating with Better Model

**Goal**: Replace poor quality annotations with predictions from improved model.

```toml
[parameters]
model_file_name = "v2-improved-model"
confidence_threshold = 0.25
batch_size = 8
image_size = 640
device = "cuda:0"
label_matching_strategy = "add"
replace_annotations = true
agnostic_nms = true
```

**⚠️ Why these settings**:
- `replace_annotations = true` - removes old annotations
- Use improved model weights
- Standard confidence for balanced results

**⚠️ Safety checklist before running**:
- [ ] Backup dataset or create new version
- [ ] Verify new model is actually better
- [ ] Test on small subset first
- [ ] Confirm you want to delete existing work

---

### Example 6: Specific Class Pre-Annotation

**Goal**: Only annotate specific classes that exist in dataset, ignore others.

```toml
[parameters]
model_file_name = "multi-class-model"
confidence_threshold = 0.3
batch_size = 8
image_size = 640
device = "cuda:0"
label_matching_strategy = "skip"
replace_annotations = false
agnostic_nms = true
```

**Example scenario**:
```
Model can detect: car, truck, bus, person, bicycle, motorcycle
Dataset only has: car, truck, bus

Result: Only "car", "truck", "bus" will be annotated
```

**Use case**:
- Using general-purpose model on specialized dataset
- Controlled label vocabulary
- Filtering unwanted classes

---

## 🔧 Parameter Tuning Workflow

### Step 1: Test on Small Sample

Always test on 10-20 images first:

```toml
[parameters]
confidence_threshold = 0.25
batch_size = 4
image_size = 640
```

### Step 2: Review Results

Check in Picsellia:
- Are objects being detected?
- Too many false positives?
- Missing obvious objects?
- Label names correct?

### Step 3: Adjust Parameters

| Observation | Action |
|-------------|--------|
| Missing many objects | Lower `confidence_threshold` (try 0.15-0.2) |
| Too many false positives | Raise `confidence_threshold` (try 0.4-0.5) |
| Wrong labels appearing | Adjust `label_matching_strategy` |
| Duplicate detections | Ensure `agnostic_nms = true` |
| Processing too slow | Increase `batch_size` or decrease `image_size` |
| Running out of memory | Decrease `batch_size` |

### Step 4: Full Dataset Processing

Once satisfied with sample results:

```toml
[parameters]
confidence_threshold = 0.25  # Your tuned value
batch_size = 16              # Increase for speed
image_size = 640
# ... other tuned parameters
```

---

## 📊 Understanding Your Results

### What Gets Created

After the pipeline completes:

1. **Bounding box annotations** on all processed images
2. **Labels** matching model predictions
3. **Confidence scores** stored with each annotation
4. **COCO JSON** file with all annotations

### Quality Indicators

**Good results**:
- Most objects detected correctly
- Few false positives
- Accurate bounding boxes
- Correct label assignment

**Needs adjustment**:
- Many missed objects → Lower confidence threshold
- Many false positives → Raise confidence threshold
- Wrong labels → Check model training or label_matching_strategy
- Duplicate boxes → Enable agnostic_nms

### Next Steps

1. **Review annotations** in Picsellia interface
2. **Correct errors** manually
3. **Accept good annotations** as-is
4. **Export dataset** for training
5. **Iterate**: Use new trained model for better pre-annotations

---

## ❓ Troubleshooting Guide

### Issue: No Annotations Created

**Check**:
1. Is `confidence_threshold` too high? Try 0.1
2. Does the model match the dataset task?
3. Are images compatible with model?
4. Check logs for errors

---

### Issue: Too Many False Positives

**Solutions**:
1. Increase `confidence_threshold` to 0.4 or 0.5
2. Verify model quality - might need retraining
3. Check if model was trained on similar data

---

### Issue: Missing Many Objects

**Solutions**:
1. Lower `confidence_threshold` to 0.15-0.2
2. Increase `image_size` to 1280 for small objects
3. Verify model can detect these object types
4. Check if objects are in model's training classes

---

### Issue: Wrong Labels on Objects

**Check**:
1. Verify model was trained correctly
2. Check `label_matching_strategy` setting
3. Review model's label mapping
4. Ensure model and dataset are compatible

---

### Issue: Duplicate/Overlapping Boxes

**Solution**:
```toml
agnostic_nms = true
```

This removes redundant overlapping detections.

---

### Issue: Pipeline Fails - "Cannot use ONNX model"

**Solution**: The model must be in PyTorch (.pt) format. ONNX models are not supported for pre-annotation.

**Fix**:
1. Use the original .pt weights instead
2. Or retrain/export model to .pt format

---

### Issue: Out of Memory

**Solutions**:
1. Reduce `batch_size` (try 4 or 2)
2. Reduce `image_size` (try 416 or 320)
3. Use smaller model variant
4. Process dataset in smaller batches

---

### Issue: Processing Too Slow

**Solutions**:
1. Increase `batch_size` (if memory allows)
2. Reduce `image_size` to 416
3. Ensure using GPU: `device = "cuda:0"`
4. Use faster model variant (YOLOv8n instead of YOLOv8x)

---

### Issue: Labels Not Being Created

**Check**:
1. Is `label_matching_strategy = "add"`?
2. If using "skip" or "strict", labels must exist in dataset first
3. Verify model has labels in metadata

---

## 💡 Best Practices

### 1. Always Test First
Run on a small subset before processing entire dataset:
- Catch configuration issues early
- Verify quality before bulk processing
- Save time and resources

### 2. Use Appropriate Confidence Threshold
- Too low (< 0.15): Waste time reviewing false positives
- Too high (> 0.5): Miss valid objects
- Sweet spot: 0.2-0.35 for most cases

### 3. Backup Before Replace
```toml
replace_annotations = true  # Only use with backups!
```
Always create a new dataset version before replacing annotations.

### 4. Match Training Image Size
If model was trained with `image_size = 640`, use the same for inference.

### 5. Review and Refine
Pre-annotations are starting points, not final truth:
1. Run pre-annotation
2. Review in Picsellia
3. Correct errors
4. Accept good annotations
5. Use refined dataset to train better model
6. Repeat

### 6. Use GPU
CPU inference is 10-20x slower than GPU. Always use:
```toml
device = "cuda:0"
```

### 7. Label Strategy Selection
- **New project**: Use `"add"` to create all labels
- **Established dataset**: Use `"skip"` to maintain label control
- **Strict requirements**: Use `"strict"` to enforce exact matches

---

## 🚀 Getting Started Checklist

- [ ] Have a trained YOLOv8 model in Picsellia
- [ ] Model weights are in .pt format (not ONNX)
- [ ] Have a dataset with images to annotate
- [ ] Decide on confidence threshold (start with 0.25)
- [ ] Choose label matching strategy
- [ ] Test on 10-20 images first
- [ ] Review test results
- [ ] Adjust parameters if needed
- [ ] Run on full dataset
- [ ] Review annotations in Picsellia
- [ ] Correct any errors manually

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- YOLOv8 model questions → See [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/)
- Pipeline configuration help → Refer to this guide

---

## 🔄 Integration with Training Pipeline

Pre-annotation and training work together in an iterative cycle:

```
1. Pre-annotate with base model (this pipeline)
   ↓
2. Review and correct annotations
   ↓
3. Train new model (YOLOv8 Training Pipeline)
   ↓
4. Pre-annotate new images with improved model
   ↓
5. Repeat - each cycle improves quality
```

This creates a virtuous cycle of continuous improvement!

---

**Pipeline Version**: 1.0.11
**Type**: Pre-Annotation
**Framework**: Ultralytics YOLOv8
**Last Updated**: 2026-01-08

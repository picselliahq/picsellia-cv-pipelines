# Grounding DINO Pre-Annotation Pipeline

**Automatically detect and annotate objects using natural language descriptions without training.**

This Picsellia pipeline uses Grounding DINO, a zero-shot object detection model that can detect any object described in plain text. Unlike traditional models that need training on specific classes, Grounding DINO understands natural language and can detect objects based on your dataset's label names.

## What You'll Get

After running this pipeline, your Picsellia dataset will contain:
- ✅ Bounding box annotations for detected objects
- ✅ Labels matching your dataset's existing label list
- ✅ Zero-shot detection (no training required)
- ✅ COCO-format annotations ready for review

## Quick Start Guide

### 🎯 Basic Pre-Annotation

**Goal**: Automatically annotate images using your dataset's labels.

```toml
[parameters]
box_threshold = 0.35
text_threshold = 0.25
```

The pipeline automatically uses your dataset's labels as text prompts for detection.

### 🎯 High-Precision Annotations

**Goal**: Only keep very confident detections.

```toml
[parameters]
box_threshold = 0.5
text_threshold = 0.35
```

Higher thresholds mean fewer but more accurate annotations.

### 🎯 High-Recall Annotations

**Goal**: Detect as many objects as possible, including uncertain ones.

```toml
[parameters]
box_threshold = 0.25
text_threshold = 0.2
```

Lower thresholds capture more objects but may include false positives.

---

## 📋 Complete Parameter Reference

### 🔵 Detection Thresholds

#### `box_threshold`
**What it does**: Minimum confidence score for a bounding box to be considered valid.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.35`
**Recommended range**: `0.25` to `0.5`

**How it works**:
- **Lower values (0.2-0.3)**: More detections, including uncertain ones
  - ✅ Higher recall - catch more objects
  - ❌ More false positives
  
- **Medium values (0.35-0.4)**: Balanced (recommended)
  - ✅ Good trade-off between precision and recall
  
- **Higher values (0.5-0.6)**: Fewer, high-confidence detections
  - ✅ Higher precision - more accurate
  - ❌ May miss some valid objects

**Visual guide**:
```
box_threshold = 0.25  →  📦📦📦📦📦  (Many detections)
box_threshold = 0.35  →  📦📦📦      (Balanced)
box_threshold = 0.50  →  📦          (Few, confident)
```

**When to adjust**:
- **Missing obvious objects?** → Lower to 0.25-0.3
- **Too many false positives?** → Raise to 0.45-0.5
- **Dense, cluttered scenes?** → Use lower (0.3)
- **Clean, simple images?** → Can use higher (0.4-0.5)

**Example**:
```toml
# High recall for difficult scenes
box_threshold = 0.25

# Balanced (recommended)
box_threshold = 0.35

# High precision
box_threshold = 0.50
```

---

#### `text_threshold`
**What it does**: Minimum confidence score for the text-to-object matching.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.25`
**Recommended range**: `0.2` to `0.4`

**How it works**: This threshold determines how confident the model must be that the detected object matches the text description (label name).

**Relationship with box_threshold**:
- `box_threshold`: "Is there an object here?"
- `text_threshold`: "Does this object match the label description?"

**When to adjust**:
- **Wrong labels on objects?** → Raise text_threshold
- **Objects detected but not labeled?** → Lower text_threshold
- **Ambiguous label names?** → Raise text_threshold

**Example**:
```toml
# More lenient text matching
text_threshold = 0.2

# Balanced (recommended)
text_threshold = 0.25

# Strict text matching
text_threshold = 0.35
```

**Best practice**: Usually keep `text_threshold` slightly lower than `box_threshold`:
```toml
box_threshold = 0.35
text_threshold = 0.25
```

---

## 🎓 Real-World Configuration Examples

### Example 1: General Purpose Pre-Annotation

**Goal**: Annotate diverse objects in various scenes.

```toml
[parameters]
box_threshold = 0.35
text_threshold = 0.25
```

**Why these settings**:
- Default balanced thresholds
- Works well for most use cases
- Good starting point

**Dataset labels example**: `["person", "car", "bicycle", "dog"]`

---

### Example 2: Difficult Detection Scenarios

**Goal**: Detect small or partially occluded objects.

```toml
[parameters]
box_threshold = 0.25
text_threshold = 0.2
```

**Why these settings**:
- Lower thresholds for higher recall
- Catches challenging cases
- Accept some false positives for review

**Use cases**:
- Crowded scenes
- Small objects
- Partial occlusions
- Low-quality images

---

### Example 3: High-Quality Annotations

**Goal**: Generate very accurate annotations needing minimal review.

```toml
[parameters]
box_threshold = 0.5
text_threshold = 0.35
```

**Why these settings**:
- High precision focus
- Fewer annotations but more accurate
- Less manual correction needed

**Use cases**:
- Clean, high-quality images
- Well-defined objects
- When false positives are costly

---

### Example 4: Specific Domain Objects

**Goal**: Detect specialized objects with clear names.

```toml
[parameters]
box_threshold = 0.4
text_threshold = 0.3
```

**Dataset labels example**: `["safety helmet", "reflective vest", "construction vehicle"]`

**Why these settings**:
- Moderate thresholds for specific objects
- Clear label names help model understand
- Balanced approach

---

### Example 5: Ambiguous Categories

**Goal**: Handle labels that might be confused or overlap.

```toml
[parameters]
box_threshold = 0.4
text_threshold = 0.35
```

**Dataset labels example**: `["car", "vehicle", "automobile"]` (overlapping concepts)

**Why these settings**:
- Higher text threshold to reduce confusion
- Strict matching when labels are similar
- Helps avoid duplicate detections

---

## 🎯 How Grounding DINO Works

### Label-Based Detection

1. **Reads your dataset labels**: Automatically uses labels like `["person", "car", "dog"]`
2. **Converts to text prompts**: Each label becomes a detection query
3. **Zero-shot detection**: Finds objects matching the text descriptions
4. **Creates annotations**: Generates bounding boxes with appropriate labels

### No Training Required

Unlike YOLOv8 or RT-DETR:
- ✅ No need for training data
- ✅ Works immediately with any label names
- ✅ Can detect novel object categories
- ✅ Understands natural language descriptions

### Best Label Naming Practices

**Good label names**:
- ✅ Simple and clear: `"person"`, `"car"`, `"dog"`
- ✅ Common objects: `"bottle"`, `"chair"`, `"laptop"`
- ✅ Specific when needed: `"red car"`, `"baseball bat"`

**Avoid**:
- ❌ Too generic: `"object"`, `"thing"`
- ❌ Too technical: `"obj_class_01"`
- ❌ Ambiguous: `"item"`

**Example label sets**:

```toml
# Good - Retail
["product", "package", "box", "bottle", "can"]

# Good - Safety
["person", "hard hat", "safety vest", "safety cone"]

# Good - Traffic
["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]

# Avoid - Too vague
["object1", "object2", "thing"]
```

---

## 📊 Understanding Your Results

### What Gets Created

After the pipeline completes:

1. **Bounding box annotations** on all images
2. **Labels** matching your dataset's label list
3. **Confidence scores** for each detection
4. **COCO JSON** with all annotations

### Quality Check

Review in Picsellia:
- Are objects correctly detected?
- Are labels accurate?
- Too many false positives?
- Missing obvious objects?

### Expected Performance

Grounding DINO performance varies by:
- **Label clarity**: Clear names → better results
- **Object visibility**: Clear, unoccluded → better detection
- **Image quality**: High resolution → better detection
- **Object size**: Larger objects → easier to detect

---

## ❓ Troubleshooting Guide

### Issue: No Annotations Created

**Check**:
1. Does your dataset have labels defined?
2. Are thresholds too high? Try lowering both to 0.2
3. Are label names too generic? Use more specific names
4. Check pipeline logs for errors

---

### Issue: Too Many False Positives

**Solutions**:
1. Increase `box_threshold` to 0.45-0.5
2. Increase `text_threshold` to 0.3-0.35
3. Review label names - ensure they're specific
4. Accept that some false positives are normal for review

---

### Issue: Missing Many Objects

**Solutions**:
1. Lower `box_threshold` to 0.25-0.3
2. Lower `text_threshold` to 0.2
3. Check if label names accurately describe objects
4. Verify objects are visible in images

---

### Issue: Wrong Labels on Objects

**Check**:
1. Are label names ambiguous? ("vehicle" vs "car" vs "automobile")
2. Increase `text_threshold` to 0.35
3. Rename labels to be more specific
4. Remove overlapping label categories

---

### Issue: Inconsistent Results

**Causes**:
- Label names vary in specificity
- Image quality varies across dataset
- Objects have different sizes/visibility

**Solutions**:
1. Standardize label names
2. Use moderate thresholds (0.35/0.25)
3. Review and correct annotations manually

---

## 💡 Best Practices

### 1. Use Clear, Descriptive Labels

```toml
# Good
Dataset labels: ["person", "car", "traffic light"]

# Better
Dataset labels: ["pedestrian", "passenger car", "traffic signal"]

# Avoid
Dataset labels: ["obj1", "thing", "item"]
```

### 2. Start with Defaults

```toml
box_threshold = 0.35
text_threshold = 0.25
```

Then adjust based on results.

### 3. Review and Refine

Grounding DINO is a starting point:
1. Run pre-annotation
2. Review in Picsellia
3. Correct errors manually
4. Accept good annotations
5. Use refined dataset for training

### 4. Test on Small Batch First

Always run on 10-20 images first:
- Verify detections are reasonable
- Check label accuracy
- Adjust thresholds if needed
- Then process full dataset

### 5. Combine with Other Pipelines

**Workflow**:
1. **Grounding DINO**: Bootstrap initial annotations
2. **Manual review**: Correct errors
3. **YOLOv8 Training**: Train custom model
4. **YOLOv8 Pre-Annotation**: Use trained model for more data

---

## 🚀 Getting Started Checklist

- [ ] Have dataset with images in Picsellia
- [ ] Define clear, descriptive labels
- [ ] Start with default thresholds (0.35/0.25)
- [ ] Test on 10-20 sample images
- [ ] Review results for quality
- [ ] Adjust thresholds if needed
- [ ] Process full dataset
- [ ] Review and correct annotations in Picsellia
- [ ] Use annotated data for training

---

## 🔗 Related Pipelines

- **YOLOv8 Pre-Annotation**: Pre-annotate with trained model
- **SAM3_Bbox**: Zero-shot detection with segmentation
- **YOLOv8 Training**: Train custom model on annotations

---

## 🎯 Grounding DINO vs Other Pre-Annotation

| Feature | Grounding DINO | YOLOv8 Pre-Annotation | SAM3 |
|---------|----------------|----------------------|------|
| **Training required** | ❌ No | ✅ Yes | ❌ No |
| **Custom objects** | ✅ Any described in text | ❌ Only trained classes | ✅ Any |
| **Setup time** | Minutes | Hours (train first) | Minutes |
| **Accuracy** | Good | Excellent (if trained well) | Good |
| **Best for** | Quick bootstrapping | Production use | Segmentation masks |
| **Input** | Label names | Trained model | Text prompts |

**Use Grounding DINO when**:
- Starting a new annotation project
- No trained model available
- Need quick initial annotations
- Have clear label names

**Use YOLOv8 Pre-Annotation when**:
- Have trained model already
- Need highest accuracy
- Specific domain/objects

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- Grounding DINO questions → See [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- Pipeline configuration help → Refer to this guide

---

## 🌟 Key Advantages

1. **Zero-shot capability**: No training required
2. **Natural language**: Use plain text to describe objects
3. **Flexible**: Works with any object categories
4. **Fast setup**: Minutes to start annotating
5. **Bootstrapping**: Perfect for starting new projects

---

**Pipeline Version**: 1.0
**Type**: Pre-Annotation
**Framework**: Grounding DINO
**Last Updated**: 2026-01-08

# Albumentations Processing Pipeline

**Augment your dataset with rotation, scaling, and noise to increase training data diversity.**

This Picsellia pipeline applies data augmentation transformations to your images and annotations using the Albumentations library. Perfect for expanding small datasets, adding variation, or creating robust training sets that generalize better.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A new dataset version with augmented images
- ✅ Properly transformed annotations (bounding boxes, masks)
- ✅ Increased dataset size and diversity
- ✅ Images with rotation, scaling, and optional noise

## Quick Start Guide

### 🎯 Basic Augmentation

**Goal**: Apply moderate rotation and scaling to expand your dataset.

```toml
[parameters]
rotate_min = -45
rotate_max = 45
scale_min = 0.9
scale_max = 1.1
rotate_prob = 1.0
add_noise = false
```

Every image will be rotated between -45° and +45° and scaled by 90-110%.

### 🎯 Light Augmentation

**Goal**: Subtle augmentation to preserve image characteristics.

```toml
[parameters]
rotate_min = -15
rotate_max = 15
scale_min = 0.95
scale_max = 1.05
rotate_prob = 1.0
add_noise = false
```

Gentle transformations for sensitive datasets.

### 🎯 Aggressive Augmentation

**Goal**: Maximum variation for small datasets.

```toml
[parameters]
rotate_min = -90
rotate_max = 90
scale_min = 0.8
scale_max = 1.2
rotate_prob = 1.0
add_noise = true
```

Strong transformations with added noise for robustness.

---

## 📋 Complete Parameter Reference

### 🔵 Rotation Parameters

#### `rotate_min`
**What it does**: Minimum rotation angle in degrees (negative = counter-clockwise).

**Type**: Integer
**Default**: `-45`
**Recommended range**: `-90` to `0`

**Example**:
```toml
# Gentle rotation
rotate_min = -15

# Moderate rotation (default)
rotate_min = -45

# Aggressive rotation
rotate_min = -90
```

---

#### `rotate_max`
**What it does**: Maximum rotation angle in degrees (positive = clockwise).

**Type**: Integer
**Default**: `45`
**Recommended range**: `0` to `90`

**Example**:
```toml
# Gentle rotation
rotate_max = 15

# Moderate rotation (default)
rotate_max = 45

# Aggressive rotation
rotate_max = 90
```

**Visual guide**:
```
rotate_min = -15, rotate_max = 15  →  slight tilt
rotate_min = -45, rotate_max = 45  →  moderate rotation
rotate_min = -90, rotate_max = 90  →  any orientation
```

---

#### `rotate_prob`
**What it does**: Probability of applying rotation (0.0 = never, 1.0 = always).

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `1.0`

**Example**:
```toml
# Always rotate
rotate_prob = 1.0

# Rotate 50% of images
rotate_prob = 0.5

# Never rotate (scale only)
rotate_prob = 0.0
```

---

### 🔵 Scale Parameters

#### `scale_min`
**What it does**: Minimum scale factor (< 1.0 = shrink, > 1.0 = enlarge).

**Type**: Float
**Default**: `0.9`
**Recommended range**: `0.7` to `1.0`

**How it works**:
- `scale_min = 1.0`: No shrinking
- `scale_min = 0.9`: Shrink to 90% of original size
- `scale_min = 0.8`: Shrink to 80% of original size

**Example**:
```toml
# Minimal scaling
scale_min = 0.95

# Moderate scaling (default)
scale_min = 0.9

# Aggressive scaling
scale_min = 0.7
```

---

#### `scale_max`
**What it does**: Maximum scale factor (< 1.0 = shrink, > 1.0 = enlarge).

**Type**: Float
**Default**: `1.1`
**Recommended range**: `1.0` to `1.3`

**How it works**:
- `scale_max = 1.0`: No enlarging
- `scale_max = 1.1`: Enlarge to 110% of original size
- `scale_max = 1.2`: Enlarge to 120% of original size

**Example**:
```toml
# Minimal scaling
scale_max = 1.05

# Moderate scaling (default)
scale_max = 1.1

# Aggressive scaling
scale_max = 1.3
```

**Note**: Images are randomly scaled between `scale_min` and `scale_max`.

---

### 🔵 Noise Parameters

#### `add_noise`
**What it does**: Whether to add random noise to images.

**Type**: Boolean
**Default**: `false`

**Why use noise**: Makes models more robust to image quality variations and sensor noise.

**Example**:
```toml
# No noise (default)
add_noise = false

# Add noise for robustness
add_noise = true
```

**When to use**:
- ✅ Small dataset needing more variation
- ✅ Training models for real-world noisy images
- ✅ Medical imaging with sensor noise
- ❌ High-quality product photography
- ❌ Clean, controlled environments

---

### 🔵 Storage Parameters

#### `datalake`
**What it does**: Name of the datalake to store augmented images.

**Type**: String
**Default**: `"default"`

**Example**:
```toml
datalake = "default"

# Custom datalake
datalake = "augmented-data"
```

---

#### `data_tag`
**What it does**: Tag to apply to augmented images.

**Type**: String
**Default**: `"processed"`

**Example**:
```toml
data_tag = "processed"

# Custom tag
data_tag = "augmented_v1"
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Small Dataset Expansion

**Goal**: Triple dataset size with diverse augmentations.

```toml
[parameters]
rotate_min = -60
rotate_max = 60
scale_min = 0.85
scale_max = 1.15
rotate_prob = 1.0
add_noise = true
datalake = "default"
data_tag = "augmented"
```

**Why these settings**:
- Wide rotation range for diversity
- Significant scaling variation
- Noise adds extra variation
- Every image gets augmented

**Use case**: < 500 images, need 3x expansion

---

### Example 2: Medical Imaging

**Goal**: Gentle augmentation preserving diagnostic features.

```toml
[parameters]
rotate_min = -10
rotate_max = 10
scale_min = 0.95
scale_max = 1.05
rotate_prob = 1.0
add_noise = true
datalake = "default"
data_tag = "medical_augmented"
```

**Why these settings**:
- Minimal rotation (medical images are orientation-sensitive)
- Subtle scaling
- Noise simulates equipment variation
- Conservative to preserve medical features

---

### Example 3: Aerial/Satellite Imagery

**Goal**: Full rotation since orientation doesn't matter.

```toml
[parameters]
rotate_min = -180
rotate_max = 180
scale_min = 0.9
scale_max = 1.1
rotate_prob = 1.0
add_noise = false
datalake = "default"
data_tag = "aerial_augmented"
```

**Why these settings**:
- Any rotation acceptable (satellite view)
- Moderate scaling
- No noise (high-quality satellite images)

---

### Example 4: Product Photography

**Goal**: Minimal augmentation for clean product images.

```toml
[parameters]
rotate_min = -5
rotate_max = 5
scale_min = 0.98
scale_max = 1.02
rotate_prob = 0.5
add_noise = false
datalake = "default"
data_tag = "product_augmented"
```

**Why these settings**:
- Very gentle transformations
- Only 50% of images rotated
- Preserve product appearance
- No noise (studio quality)

---

### Example 5: Street Scene / Autonomous Driving

**Goal**: Realistic variations for vehicle detection.

```toml
[parameters]
rotate_min = -15
rotate_max = 15
scale_min = 0.9
scale_max = 1.1
rotate_prob = 1.0
add_noise = true
datalake = "default"
data_tag = "street_augmented"
```

**Why these settings**:
- Limited rotation (cars don't rotate much)
- Moderate scaling (camera distance varies)
- Noise simulates different cameras/conditions

---

### Example 6: Document Processing

**Goal**: Simulate scanning variations.

```toml
[parameters]
rotate_min = -5
rotate_max = 5
scale_min = 0.95
scale_max = 1.05
rotate_prob = 1.0
add_noise = true
datalake = "default"
data_tag = "document_augmented"
```

**Why these settings**:
- Small rotation (documents slightly tilted)
- Light scaling
- Noise simulates scanner/camera artifacts

---

## 📊 Understanding Your Results

### What Gets Created

After the pipeline completes:

1. **New dataset version** with augmented images
2. **Transformed images** with rotation and scaling applied
3. **Adjusted annotations** (bounding boxes, polygons repositioned)
4. **Same label structure** as original dataset

### Annotation Handling

The pipeline correctly transforms:
- ✅ Bounding boxes (repositioned and resized)
- ✅ Polygon segmentation masks (rotated and scaled)
- ✅ Keypoints (if present)

### Dataset Size Impact

```
Original dataset: 1,000 images
After augmentation: 1,000 new images
Total available: 2,000 images (original + augmented)
```

You can combine both for training or use augmented data separately.

---

## ❓ Troubleshooting Guide

### Issue: Annotations Don't Match Images

**Cause**: Rare edge case with extreme transformations.

**Solutions**:
1. Reduce rotation range (try ±45° max)
2. Reduce scaling range (0.9-1.1)
3. Verify original annotations are correct
4. Check pipeline logs for errors

---

### Issue: Images Look Over-Augmented

**Cause**: Too aggressive parameters.

**Solutions**:
1. Reduce rotation range
2. Reduce scaling range
3. Set `add_noise = false`
4. Use more conservative settings

---

### Issue: Not Enough Variation

**Cause**: Too conservative parameters.

**Solutions**:
1. Increase rotation range
2. Increase scaling range
3. Set `add_noise = true`
4. Consider running pipeline multiple times with different settings

---

### Issue: Pipeline Runs But No Images Created

**Check**:
1. Are parameters within valid ranges?
2. Is datalake accessible?
3. Check pipeline logs
4. Verify input dataset has images

---

## 💡 Best Practices

### 1. Match Augmentation to Your Domain

```toml
# Aerial imagery - any rotation OK
rotate_min = -180, rotate_max = 180

# Street scenes - limited rotation
rotate_min = -15, rotate_max = 15

# Medical imaging - minimal augmentation
rotate_min = -10, rotate_max = 10
```

### 2. Don't Over-Augment

Too much augmentation can:
- Make images unrealistic
- Confuse the model
- Reduce training effectiveness

**Rule of thumb**: Augmented images should still look natural.

### 3. Combine with Other Augmentation

This pipeline handles:
- Rotation
- Scaling  
- Noise

For more augmentation, use during training:
- Color jittering
- Brightness/contrast
- Flipping
- Cropping

### 4. Preview Before Full Processing

Run on 10-20 images first:
- Verify augmentations look good
- Check annotations are correct
- Adjust parameters if needed
- Then process full dataset

### 5. Tag Augmented Data

```toml
data_tag = "augmented_rotation_45"
```

Clear tags help organize and track different augmentation strategies.

---

## 🚀 Getting Started Checklist

- [ ] Have annotated dataset in Picsellia
- [ ] Decide on rotation range based on domain
- [ ] Set appropriate scaling range
- [ ] Decide if noise is beneficial
- [ ] Test on 10-20 sample images
- [ ] Review augmented results
- [ ] Adjust parameters if needed
- [ ] Process full dataset
- [ ] Combine with original for training

---

## 🔗 Related Pipelines

- **YOLOv8 Training**: Train on augmented dataset
- **Dataset Tiler**: Another preprocessing approach
- **Bounding Box Cropper**: Extract object crops

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- Albumentations library → See [Albumentations docs](https://albumentations.ai/)
- Pipeline configuration help → Refer to this guide

---

## 🎯 Key Advantages

1. **Increase dataset size**: Turn 100 images into 200+
2. **Improve model robustness**: Handle variations better
3. **Preserve annotations**: Automatic transformation of bounding boxes and masks
4. **Simple configuration**: Just a few parameters to set
5. **Production-ready**: Battle-tested Albumentations library

---

**Pipeline Version**: 1.0.2
**Type**: Dataset Version Creation
**Supported Types**: Object Detection, Segmentation
**Last Updated**: 2026-01-08

# YOLOv8 Training Pipeline

**Train state-of-the-art YOLOv8 object detection models on your custom datasets.**

This Picsellia pipeline trains Ultralytics YOLOv8 models for object detection tasks. Starting from pretrained weights, the pipeline fine-tunes the model on your annotated dataset, applies data augmentation, and exports the trained model ready for deployment.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A fully trained YOLOv8 model fine-tuned on your data
- ✅ Model weights saved as experiment artifacts
- ✅ Exported model in your chosen format (ONNX, TorchScript, etc.)
- ✅ Evaluation metrics on your test set
- ✅ Training curves and visualizations in Picsellia

## Quick Start Guide

### 🎯 Basic Training

**Goal**: Train a YOLOv8 model with default settings.

```toml
[hyperparameters]
epochs = 100
batch_size = 16
image_size = 640
device = "cuda:0"
```

This will train for 100 epochs with reasonable defaults for most use cases.

### 🎯 Fast Training for Testing

**Goal**: Quick training run to validate your setup.

```toml
[hyperparameters]
epochs = 10
batch_size = 8
image_size = 640
device = "cuda:0"
patience = 5
```

Shorter training with early stopping for rapid iteration.

### 🎯 Production Training

**Goal**: High-quality model with extensive augmentation.

```toml
[hyperparameters]
epochs = 300
batch_size = 16
image_size = 640
device = "cuda:0"
patience = 50

# Augmentation
mosaic = 1.0
mixup = 0.15
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
degrees = 10.0
translate = 0.1
scale = 0.9
fliplr = 0.5
```

Comprehensive training configuration for best model performance.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Training Parameters

#### `epochs`
**What it does**: Number of complete passes through the training dataset.

**Type**: Integer
**Default**: `100`
**Recommended range**: `50` to `300`

**How to choose**:
- **Quick experiments**: 10-30 epochs
- **Standard training**: 100-150 epochs
- **Production models**: 200-300 epochs

**Example**:
```toml
epochs = 150
```

**Note**: Use `patience` parameter for early stopping if the model stops improving.

---

#### `batch_size`
**What it does**: Number of images processed together in one training step.

**Type**: Integer
**Default**: `16`
**Recommended range**: `8` to `64`

**How it affects training**:
- **Larger batch (32-64)**: 
  - ✅ Faster training
  - ✅ More stable gradients
  - ❌ Requires more GPU memory
  
- **Smaller batch (8-16)**: 
  - ✅ Works with limited GPU memory
  - ✅ Can improve generalization
  - ❌ Slower training
  - ❌ Noisier gradients

**GPU Memory Guide**:
```
4 GB VRAM  → batch_size = 4-8
8 GB VRAM  → batch_size = 8-16
12 GB VRAM → batch_size = 16-32
24 GB VRAM → batch_size = 32-64
```

**Example**:
```toml
batch_size = 16
```

---

#### `image_size`
**What it does**: Size to which all images are resized during training.

**Type**: Integer
**Default**: `640`
**Common values**: `320`, `416`, `640`, `1280`

**Trade-offs**:
- **Smaller (320-416)**: Faster training, less detail
- **Medium (640)**: Balanced performance (recommended)
- **Larger (1280)**: Better small object detection, slower training

**When to adjust**:
- Small objects to detect? → Use 1280
- Fast inference needed? → Use 416 or 640
- Limited GPU memory? → Use 320 or 416

**Example**:
```toml
image_size = 640
```

---

#### `device`
**What it does**: Hardware device for training.

**Type**: String
**Options**: `"cuda:0"`, `"cuda:1"`, `"cpu"`, `"mps"`

**Example**:
```toml
# Use first GPU
device = "cuda:0"

# Use CPU (very slow, not recommended)
device = "cpu"

# Use second GPU
device = "cuda:1"

# Use Apple Metal (Mac M1/M2)
device = "mps"
```

---

### 🔵 Training Control Parameters

#### `patience`
**What it does**: Number of epochs to wait for improvement before stopping early.

**Type**: Integer
**Default**: `100`
**Recommended range**: `20` to `100`

**How it works**: If validation metrics don't improve for `patience` epochs, training stops automatically.

**Example**:
```toml
# Stop if no improvement for 50 epochs
patience = 50

# Disable early stopping
patience = 0
```

---

#### `save_period`
**What it does**: Save model checkpoint every N epochs.

**Type**: Integer
**Default**: `100`

**Example**:
```toml
# Save checkpoint every 10 epochs
save_period = 10
```

---

#### `close_mosaic`
**What it does**: Disable mosaic augmentation in the last N epochs.

**Type**: Integer
**Default**: `10`

**Why it matters**: Disabling mosaic near the end helps the model learn on non-augmented images for better real-world performance.

**Example**:
```toml
close_mosaic = 10
```

---

#### `seed`
**What it does**: Random seed for reproducible results.

**Type**: Integer
**Default**: `0`

**Example**:
```toml
seed = 42
```

---

#### `validate`
**What it does**: Run validation during training.

**Type**: Boolean
**Default**: `true`

**Example**:
```toml
validate = true
```

---

#### `train_set_split_ratio`
**What it does**: Fraction of data to use for training (rest used for validation).

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.8`

**Example**:
```toml
# 80% train, 20% validation
train_set_split_ratio = 0.8

# 90% train, 10% validation
train_set_split_ratio = 0.9
```

---

### 🔵 Optimizer Parameters

#### `optimizer`
**What it does**: Optimization algorithm to use.

**Type**: String
**Default**: `"auto"`
**Options**: `"auto"`, `"SGD"`, `"Adam"`, `"AdamW"`, `"RMSProp"`

**Example**:
```toml
optimizer = "auto"
```

---

#### `lr0`
**What it does**: Initial learning rate.

**Type**: Float
**Default**: `0.01`
**Recommended range**: `0.001` to `0.1`

**Example**:
```toml
lr0 = 0.01
```

---

#### `lrf`
**What it does**: Final learning rate multiplier (final_lr = lr0 * lrf).

**Type**: Float
**Default**: `0.1`

**Example**:
```toml
lrf = 0.1
```

---

#### `momentum`
**What it does**: SGD momentum/Adam beta1.

**Type**: Float
**Default**: `0.937`
**Range**: `0.0` to `1.0`

**Example**:
```toml
momentum = 0.937
```

---

#### `weight_decay`
**What it does**: L2 regularization coefficient.

**Type**: Float
**Default**: `0.0005`

**Example**:
```toml
weight_decay = 0.0005
```

---

### 🔵 Learning Rate Scheduling

#### `warmup_epochs`
**What it does**: Number of epochs for learning rate warmup.

**Type**: Float
**Default**: `3.0`

**Example**:
```toml
warmup_epochs = 3.0
```

---

#### `warmup_momentum`
**What it does**: Initial momentum during warmup.

**Type**: Float
**Default**: `0.8`

**Example**:
```toml
warmup_momentum = 0.8
```

---

#### `warmup_bias_lr`
**What it does**: Learning rate for bias parameters during warmup.

**Type**: Float
**Default**: `0.1`

**Example**:
```toml
warmup_bias_lr = 0.1
```

---

#### `cos_lr`
**What it does**: Use cosine learning rate scheduler.

**Type**: Boolean
**Default**: `false`

**Example**:
```toml
cos_lr = true
```

---

### 🔵 Loss Function Weights

#### `box`
**What it does**: Weight for bounding box loss.

**Type**: Float
**Default**: `7.5`

**Example**:
```toml
box = 7.5
```

---

#### `cls`
**What it does**: Weight for classification loss.

**Type**: Float
**Default**: `0.5`

**Example**:
```toml
cls = 0.5
```

---

#### `dfl`
**What it does**: Weight for distribution focal loss.

**Type**: Float
**Default**: `1.5`

**Example**:
```toml
dfl = 1.5
```

---

### 🔵 Data Augmentation Parameters

#### Color Augmentations

##### `hsv_h`
**What it does**: Hue augmentation range.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.015`

**Example**:
```toml
hsv_h = 0.015
```

---

##### `hsv_s`
**What it does**: Saturation augmentation range.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.7`

**Example**:
```toml
hsv_s = 0.7
```

---

##### `hsv_v`
**What it does**: Value (brightness) augmentation range.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.4`

**Example**:
```toml
hsv_v = 0.4
```

---

#### Geometric Augmentations

##### `degrees`
**What it does**: Random rotation range in degrees.

**Type**: Float
**Range**: `-180.0` to `180.0`
**Default**: `0.0`

**Example**:
```toml
# Rotate up to ±10 degrees
degrees = 10.0
```

---

##### `translate`
**What it does**: Random translation as fraction of image size.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.1`

**Example**:
```toml
translate = 0.1
```

---

##### `scale`
**What it does**: Random scale range.

**Type**: Float
**Default**: `0.5`

**Example**:
```toml
scale = 0.5
```

---

##### `shear`
**What it does**: Random shear angle in degrees.

**Type**: Float
**Range**: `-180.0` to `180.0`
**Default**: `0.0`

**Example**:
```toml
shear = 5.0
```

---

##### `perspective`
**What it does**: Random perspective transformation.

**Type**: Float
**Range**: `0.0` to `0.001`
**Default**: `0.0`

**Example**:
```toml
perspective = 0.0001
```

---

##### `flipud`
**What it does**: Probability of vertical flip.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.0`

**Example**:
```toml
flipud = 0.5
```

---

##### `fliplr`
**What it does**: Probability of horizontal flip.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.5`

**Example**:
```toml
fliplr = 0.5
```

---

#### Advanced Augmentations

##### `mosaic`
**What it does**: Probability of mosaic augmentation (combines 4 images).

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `1.0`

**Example**:
```toml
mosaic = 1.0
```

---

##### `mixup`
**What it does**: Probability of mixup augmentation (blends 2 images).

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.0`

**Example**:
```toml
mixup = 0.15
```

---

##### `copy_paste`
**What it does**: Probability of copy-paste augmentation.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.0`

**Example**:
```toml
copy_paste = 0.3
```

---

##### `auto_augment`
**What it does**: AutoAugment policy.

**Type**: String
**Default**: `"randaugment"`
**Options**: `"randaugment"`, `"autoaugment"`, `"augmix"`

**Example**:
```toml
auto_augment = "randaugment"
```

---

##### `erasing`
**What it does**: Probability of random erasing.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `0.4`

**Example**:
```toml
erasing = 0.4
```

---

##### `crop_fraction`
**What it does**: Fraction of image to crop during classification training.

**Type**: Float
**Range**: `0.1` to `1.0`
**Default**: `1.0`

**Example**:
```toml
crop_fraction = 1.0
```

---

### 🔵 Advanced Training Options

#### `workers`
**What it does**: Number of data loading workers.

**Type**: Integer
**Default**: `8`

**Example**:
```toml
workers = 8
```

---

#### `cache`
**What it does**: Cache images in memory for faster training.

**Type**: Boolean
**Default**: `false`

**Example**:
```toml
cache = true
```

**Warning**: Requires significant RAM for large datasets.

---

#### `deterministic`
**What it does**: Enable deterministic training for reproducibility.

**Type**: Boolean
**Default**: `true`

**Example**:
```toml
deterministic = true
```

---

#### `amp`
**What it does**: Use Automatic Mixed Precision for faster training.

**Type**: Boolean
**Default**: `true`

**Example**:
```toml
amp = true
```

---

#### `fraction`
**What it does**: Fraction of dataset to use for training.

**Type**: Float
**Range**: `0.0` to `1.0`
**Default**: `1.0`

**Example**:
```toml
# Use only 50% of data
fraction = 0.5
```

---

#### `freeze`
**What it does**: Number of layers to freeze during training.

**Type**: Integer (optional)
**Default**: `None`

**Example**:
```toml
# Freeze first 10 layers
freeze = 10
```

---

#### `label_smoothing`
**What it does**: Label smoothing epsilon.

**Type**: Float
**Default**: `0.0`

**Example**:
```toml
label_smoothing = 0.1
```

---

#### `dropout`
**What it does**: Dropout rate for regularization.

**Type**: Float
**Default**: `0.0`

**Example**:
```toml
dropout = 0.1
```

---

### 🔵 Export Parameters

#### `export_format`
**What it does**: Format for model export.

**Type**: String
**Default**: `"onnx"`
**Options**: `"onnx"`, `"torchscript"`, `"coreml"`, `"tensorflow"`, `"tflite"`

**Example**:
```toml
export_format = "onnx"
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Small Dataset (< 1000 images)

```toml
[hyperparameters]
epochs = 150
batch_size = 16
image_size = 640
device = "cuda:0"
patience = 30

# Aggressive augmentation
mosaic = 1.0
mixup = 0.2
copy_paste = 0.1
hsv_h = 0.02
hsv_s = 0.7
hsv_v = 0.4
degrees = 15.0
translate = 0.2
scale = 0.9
fliplr = 0.5
```

**Why these settings**:
- Aggressive augmentation prevents overfitting on small datasets
- Moderate epoch count with early stopping
- Standard batch size

---

### Example 2: Large Dataset (> 10,000 images)

```toml
[hyperparameters]
epochs = 100
batch_size = 32
image_size = 640
device = "cuda:0"
patience = 50
cache = true

# Moderate augmentation
mosaic = 1.0
mixup = 0.1
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
degrees = 5.0
fliplr = 0.5
```

**Why these settings**:
- Larger batch size for faster training
- Less aggressive augmentation (data is plentiful)
- Cache enabled for speed (if RAM allows)

---

### Example 3: Small Object Detection

```toml
[hyperparameters]
epochs = 200
batch_size = 8
image_size = 1280
device = "cuda:0"
patience = 50

# Light augmentation to preserve small objects
mosaic = 0.5
mixup = 0.0
scale = 0.2
translate = 0.05
degrees = 5.0
```

**Why these settings**:
- Larger image size (1280) for better small object detection
- Smaller batch size (larger images need more memory)
- Conservative augmentation to avoid losing small objects

---

### Example 4: Fast Training for Prototyping

```toml
[hyperparameters]
epochs = 50
batch_size = 16
image_size = 416
device = "cuda:0"
patience = 10
fraction = 0.5

# Minimal augmentation
mosaic = 0.5
hsv_h = 0.01
hsv_s = 0.5
hsv_v = 0.3
fliplr = 0.5
```

**Why these settings**:
- Smaller image size for speed
- Use only 50% of data
- Early stopping after 10 epochs
- Quick iteration cycles

---

### Example 5: Production Model

```toml
[hyperparameters]
epochs = 300
batch_size = 16
image_size = 640
device = "cuda:0"
patience = 100
seed = 42
deterministic = true

# Comprehensive augmentation
mosaic = 1.0
mixup = 0.15
copy_paste = 0.1
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
degrees = 10.0
translate = 0.1
scale = 0.9
shear = 2.0
perspective = 0.0001
fliplr = 0.5
erasing = 0.4

# Optimized learning
lr0 = 0.01
lrf = 0.1
momentum = 0.937
weight_decay = 0.0005
warmup_epochs = 5.0
cos_lr = true

# Export
export_format = "onnx"
```

**Why these settings**:
- Long training with high patience for best results
- Full augmentation suite
- Cosine learning rate schedule
- Deterministic for reproducibility

---

### Example 6: Transfer Learning from Scratch

```toml
[hyperparameters]
epochs = 200
batch_size = 16
image_size = 640
device = "cuda:0"

# Start with lower learning rate
lr0 = 0.001
warmup_epochs = 10.0

# Strong augmentation
mosaic = 1.0
mixup = 0.2
degrees = 20.0
translate = 0.2
```

**Why these settings**:
- Lower initial learning rate for transfer learning
- Longer warmup period
- Strong augmentation for better generalization

---

## 🔧 Parameter Tuning Workflow

### Step 1: Start with Defaults
```toml
[hyperparameters]
epochs = 100
batch_size = 16
image_size = 640
device = "cuda:0"
```

### Step 2: Monitor Training

Watch for these patterns in Picsellia:

| Observation | Action |
|-------------|--------|
| Loss decreasing steadily | Continue training |
| Loss plateaus early | Increase learning rate or augmentation |
| Loss oscillates wildly | Decrease learning rate or batch size |
| Overfitting (train << val loss) | Increase augmentation |
| Underfitting (both losses high) | Train longer or increase model size |
| OOM errors | Decrease batch size or image size |

### Step 3: Adjust Parameters

Based on results, tune key parameters:

1. **If overfitting**: Increase `mosaic`, `mixup`, `dropout`
2. **If underfitting**: Train longer (`epochs`), reduce augmentation
3. **If too slow**: Increase `batch_size`, decrease `image_size`
4. **If unstable**: Decrease `lr0`, increase `warmup_epochs`

---

## 📊 Understanding Your Results

### Training Artifacts

After training completes, check Picsellia for:

- **best-model**: Best model weights (saved automatically)
- **Training curves**: Loss, mAP, precision, recall over epochs
- **Visualizations**: Predictions on validation images
- **Metrics**: Final mAP, precision, recall on test set

### Model Export

The model is automatically exported in your chosen format:
- **ONNX**: Cross-platform deployment
- **TorchScript**: PyTorch production
- **CoreML**: iOS/macOS apps
- **TFLite**: Mobile and edge devices

---

## ❓ Troubleshooting Guide

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce `batch_size` (try 8 or 4)
2. Reduce `image_size` (try 416 or 320)
3. Disable `cache`
4. Reduce `workers`

---

### Issue: Training Too Slow

**Solutions**:
1. Increase `batch_size` (if memory allows)
2. Enable `cache = true` (if RAM allows)
3. Reduce `image_size`
4. Use fewer `workers` (paradoxically can help on some systems)
5. Ensure using GPU: `device = "cuda:0"`

---

### Issue: Model Not Learning

**Check**:
1. Is loss decreasing at all? Check learning curves
2. Try increasing `lr0` (e.g., 0.01 → 0.02)
3. Check your data quality and labels
4. Ensure sufficient training data

---

### Issue: Overfitting

**Solutions**:
1. Increase augmentation: `mixup`, `mosaic`, `copy_paste`
2. Add `dropout = 0.1`
3. Increase `weight_decay`
4. Add more training data
5. Reduce model complexity (use smaller YOLOv8 variant)

---

### Issue: Poor Performance on Small Objects

**Solutions**:
1. Increase `image_size` to 1280
2. Reduce aggressive augmentations (lower `scale`, `translate`)
3. Adjust `mosaic` to 0.5 or lower
4. Use YOLOv8x (largest model)

---

### Issue: Training Crashes

**Check**:
1. GPU drivers and CUDA version
2. Sufficient disk space for checkpoints
3. Dataset integrity (corrupted images?)
4. Try `amp = false` to disable mixed precision

---

## 🚀 Getting Started Checklist

- [ ] Prepare annotated dataset in Picsellia
- [ ] Choose base YOLOv8 model (n/s/m/l/x)
- [ ] Start with default parameters
- [ ] Run short training (10 epochs) to validate setup
- [ ] Monitor training curves in Picsellia
- [ ] Adjust parameters based on observations
- [ ] Run full training
- [ ] Evaluate on test set
- [ ] Export model for deployment

---

## 💡 Best Practices

1. **Always use GPU**: CPU training is prohibitively slow
2. **Start small**: Test with a few epochs before full training
3. **Use early stopping**: Set reasonable `patience` to avoid wasting time
4. **Monitor actively**: Check Picsellia regularly during training
5. **Version control**: Use Picsellia experiments to track different configurations
6. **Test augmentation**: Visualize augmented images to ensure they make sense
7. **Reproducibility**: Set `seed` and `deterministic = true` for consistent results

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- YOLOv8 model questions → See [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/)
- Pipeline configuration help → Refer to this guide

---

**Pipeline Version**: 1.0
**Type**: Training
**Framework**: Ultralytics YOLOv8
**Last Updated**: 2026-01-08

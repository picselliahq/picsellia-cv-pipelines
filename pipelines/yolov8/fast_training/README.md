# YOLOv8 Fast Training Pipeline

**Quickly train YOLOv8 models with sensible defaults and minimal configuration.**

This Picsellia pipeline is a streamlined version of the full YOLOv8 training pipeline, designed for rapid prototyping and testing. It uses the Ultralytics default hyperparameters and augmentation settings, requiring you to configure only the essential parameters.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A trained YOLOv8 model fine-tuned on your data
- ✅ Model weights saved as experiment artifacts
- ✅ Exported model in your chosen format
- ✅ Evaluation metrics on your test set
- ✅ Training curves in Picsellia

## When to Use This Pipeline

**Use Fast Training when**:
- 🚀 Quickly testing a new dataset
- 🚀 Prototyping and iteration speed is critical
- 🚀 Default Ultralytics settings work well for your use case
- 🚀 You don't need fine-grained control over training

**Use Regular Training when**:
- 🎯 Optimizing for production deployment
- 🎯 Need custom augmentation strategies
- 🎯 Require specific learning rate schedules
- 🎯 Need full control over all hyperparameters

## Quick Start Guide

### 🎯 Basic Fast Training

**Goal**: Train quickly with all defaults.

```toml
[hyperparameters]
epochs = 50
batch_size = 16
image_size = 640
device = "cuda:0"
```

That's it! All other parameters use Ultralytics defaults.

### 🎯 With Early Stopping

**Goal**: Stop training when model stops improving.

```toml
[hyperparameters]
epochs = 100
batch_size = 16
image_size = 640
device = "cuda:0"
patience = 20
```

Training will stop automatically if no improvement for 20 epochs.

---

## 📋 Parameter Reference

This pipeline uses a **minimal parameter set**. Most Ultralytics hyperparameters and augmentation settings use their framework defaults.

### 🔵 Configurable Parameters

#### Core Training Parameters

All standard YOLOv8 training parameters are available and use Ultralytics defaults:
- `epochs` (default: 100)
- `batch_size` (default: 16)
- `image_size` (default: 640)
- `device` (default: "cuda:0")
- `lr0` (default: 0.01)
- `seed` (default: 0)
- `validate` (default: true)
- `train_set_split_ratio` (default: 0.8)
- All optimizer parameters (momentum, weight_decay, etc.)
- All loss weights (box, cls, dfl)

#### Augmentation Parameters

All Ultralytics augmentation parameters are available with their defaults:
- `mosaic` (default: 1.0)
- `mixup` (default: 0.0)
- `hsv_h`, `hsv_s`, `hsv_v` (color augmentation)
- `degrees`, `translate`, `scale`, `shear` (geometric)
- `fliplr`, `flipud` (flipping)
- And all other Ultralytics augmentations

See the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/modes/train/) for default values.

---

#### `patience`
**What it does**: Number of epochs to wait for improvement before early stopping.

**Type**: Integer
**Default**: `100`

**Example**:
```toml
patience = 20
```

---

#### `save_period`
**What it does**: Save model checkpoint every N epochs.

**Type**: Integer
**Default**: `100`

**Example**:
```toml
save_period = 10
```

---

#### `close_mosaic`
**What it does**: Disable mosaic augmentation in the last N epochs.

**Type**: Integer
**Default**: `0`

**Example**:
```toml
close_mosaic = 10
```

---

#### `export_format`
**What it does**: Format for model export after training.

**Type**: String
**Default**: `"onnx"`
**Options**: `"onnx"`, `"torchscript"`, `"coreml"`, `"tensorflow"`, `"tflite"`

**Example**:
```toml
export_format = "onnx"
```

---

## 🎓 Configuration Examples

### Example 1: Rapid Prototyping

**Goal**: Test if dataset works with YOLOv8 in under 30 minutes.

```toml
[hyperparameters]
epochs = 20
batch_size = 16
image_size = 640
device = "cuda:0"
patience = 5
```

**Expected time**: 10-30 minutes depending on dataset size.

---

### Example 2: Standard Fast Training

**Goal**: Get a decent model quickly without tuning.

```toml
[hyperparameters]
epochs = 100
batch_size = 16
image_size = 640
device = "cuda:0"
patience = 20
close_mosaic = 10
```

**Expected time**: 1-3 hours depending on dataset size.

---

### Example 3: Limited GPU Memory

**Goal**: Train on 4-6 GB GPU.

```toml
[hyperparameters]
epochs = 100
batch_size = 4
image_size = 416
device = "cuda:0"
patience = 20
```

Reduced batch size and image size to fit in memory.

---

### Example 4: High-Quality Fast Training

**Goal**: Best results with defaults.

```toml
[hyperparameters]
epochs = 150
batch_size = 16
image_size = 640
device = "cuda:0"
patience = 50
close_mosaic = 10
export_format = "onnx"
```

Longer training with generous patience.

---

## 🔄 Migrating to Full Training Pipeline

When fast training results are promising, migrate to the full training pipeline for optimization:

**Steps**:
1. Note your fast training results (mAP, loss curves)
2. Use the full YOLOv8 Training Pipeline
3. Start with same basic parameters (epochs, batch_size, etc.)
4. Add custom augmentation strategies
5. Tune learning rates and optimizer settings
6. Iterate to improve performance

**Example migration**:

**Fast Training Config**:
```toml
[hyperparameters]
epochs = 100
batch_size = 16
image_size = 640
device = "cuda:0"
```

**Full Training Config** (same starting point):
```toml
[hyperparameters]
# Core (same as fast training)
epochs = 100
batch_size = 16
image_size = 640
device = "cuda:0"

# Now add optimizations
patience = 30
close_mosaic = 10

# Custom augmentation
mosaic = 1.0
mixup = 0.15
degrees = 10.0
translate = 0.1
scale = 0.9
fliplr = 0.5

# Learning rate tuning
lr0 = 0.01
lrf = 0.1
cos_lr = true
warmup_epochs = 5.0
```

---

## 📊 Understanding Your Results

### What to Check

After training completes, review in Picsellia:

1. **Training curves**: Is loss decreasing?
2. **Validation metrics**: mAP, precision, recall
3. **Sample predictions**: Visual quality check
4. **Final metrics**: Performance on test set

### Interpreting Results

| Result | Next Step |
|--------|-----------|
| Good performance with defaults | Use in production or fine-tune with full pipeline |
| Mediocre performance | Try full training pipeline with custom parameters |
| Poor performance | Check data quality, try different model size, or use full pipeline |
| Overfitting (train >> val) | Use full pipeline with more augmentation |
| Underfitting (both losses high) | Train longer or use larger model |

---

## ❓ Troubleshooting Guide

### Issue: Training Too Slow

**Solutions**:
1. This pipeline already uses defaults - consider hardware upgrade
2. Reduce `image_size` to 416
3. Increase `batch_size` if memory allows
4. Use YOLOv8n (nano) instead of larger variants

---

### Issue: Out of Memory

**Solutions**:
```toml
batch_size = 4       # Reduce from 16
image_size = 416     # Reduce from 640
```

---

### Issue: Not Enough Control

**Solution**: Migrate to the full YOLOv8 Training Pipeline for access to all hyperparameters and augmentation settings.

---

### Issue: Model Not Learning

**Check**:
1. Verify data quality and labels
2. Check if loss is decreasing at all
3. Try full training pipeline with custom learning rates
4. Ensure sufficient training data

---

### Issue: Want Custom Augmentation

**Solution**: This pipeline uses Ultralytics defaults. For custom augmentation, use the full YOLOv8 Training Pipeline where you can set:
- Individual augmentation parameters
- Custom augmentation strategies
- Fine-grained control over augmentation probability

---

## 💡 Best Practices

### 1. Start Here, Then Optimize
Fast training is perfect for:
- Initial dataset validation
- Model architecture selection (n/s/m/l/x)
- Baseline performance metrics

Then migrate to full pipeline for optimization.

### 2. Use for Iteration Speed
When testing multiple datasets or approaches:
- Fast training for quick feedback
- Full training for final model

### 3. Trust the Defaults
Ultralytics defaults are well-tuned for most cases. Only customize if:
- Fast training shows promise
- You need production-grade performance
- Defaults clearly don't fit your use case

### 4. Document Baseline
Record fast training results as baseline:
```
Fast Training Baseline:
- mAP50: 0.65
- mAP50-95: 0.42
- Training time: 2 hours
- Parameters: All defaults
```

Then compare against optimized full training.

---

## 🚀 Getting Started Checklist

- [ ] Prepare annotated dataset in Picsellia
- [ ] Choose YOLOv8 model variant (n/s/m/l/x)
- [ ] Set basic parameters (epochs, batch_size, device)
- [ ] Run training
- [ ] Review results in Picsellia
- [ ] Decide: deploy as-is or optimize with full pipeline

---

## 🔗 Related Pipelines

- **YOLOv8 Training**: Full-featured training with all hyperparameters
- **YOLOv8 Pre-Annotation**: Generate annotations with trained model
- **YOLOv7 Segmentation**: For instance segmentation tasks

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- YOLOv8 questions → See [Ultralytics documentation](https://docs.ultralytics.com/)
- Need more control → Use the full YOLOv8 Training Pipeline

---

## Key Differences: Fast vs Full Training

| Feature | Fast Training | Full Training |
|---------|---------------|---------------|
| **Configuration** | Minimal (4 custom params) | Complete (50+ params) |
| **Augmentation** | Ultralytics defaults | Fully customizable |
| **Learning rate** | Default schedule | Custom schedules + warmup |
| **Setup time** | < 5 minutes | 10-30 minutes |
| **Best for** | Prototyping, testing | Production, optimization |
| **Control** | Basic | Complete |

---

**Pipeline Version**: 1.0
**Type**: Training (Simplified)
**Framework**: Ultralytics YOLOv8
**Last Updated**: 2026-01-08

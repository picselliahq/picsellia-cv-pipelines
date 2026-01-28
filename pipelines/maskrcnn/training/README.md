# Mask R-CNN Training Pipeline

**Train state-of-the-art Mask R-CNN models for instance segmentation with bounding boxes and pixel-level masks.**

This Picsellia pipeline trains Mask R-CNN models using PyTorch and TorchVision. Mask R-CNN is a powerful instance segmentation model that simultaneously detects objects and generates high-quality segmentation masks, making it ideal for applications requiring precise object boundaries.

## What You'll Get

After running this pipeline, you'll have:
- A fully trained Mask R-CNN model fine-tuned on your data
- Model weights saved as experiment artifacts
- Evaluation metrics on your test set
- Training curves and visualizations in Picsellia
- Instance segmentation with both boxes and masks

## Quick Start Guide

### Basic Training

**Goal**: Train Mask R-CNN with default settings.

```toml
[hyperparameters]
epochs = 50
batch_size = 4
image_size = 800
backbone = "resnet50"
```

This will fine-tune the Mask R-CNN ResNet-50 FPN model on your dataset.

### Fast Prototyping

**Goal**: Quick training for testing.

```toml
[hyperparameters]
epochs = 10
batch_size = 2
image_size = 640
backbone = "resnet50"
```

Shorter training for rapid iteration.

---

## Complete Parameter Reference

### Essential Parameters

#### `epochs`
**What it does**: Number of complete passes through the training dataset.

**Type**: Integer
**Default**: `10`
**Recommended range**: `20` to `100`

**Example**:
```toml
# Quick training
epochs = 10

# Standard training
epochs = 50

# Production training
epochs = 100
```

---

#### `batch_size`
**What it does**: Number of images processed together in one training step.

**Type**: Integer
**Default**: `4`
**Recommended range**: `2` to `8`

**GPU Memory Guide**:
```
8 GB VRAM  -> batch_size = 1-2
16 GB VRAM -> batch_size = 2-4
24 GB VRAM -> batch_size = 4-8
```

**Example**:
```toml
batch_size = 4
```

**Note**: Mask R-CNN is memory-intensive due to mask predictions. Use smaller batch sizes than object detection models.

---

#### `image_size`
**What it does**: Minimum size to which images are resized during training.

**Type**: Integer
**Default**: `800`
**Common values**: `640`, `800`, `1024`

**Example**:
```toml
# Standard
image_size = 800

# Lower resolution for speed
image_size = 640

# Higher resolution for accuracy
image_size = 1024
```

---

#### `backbone`
**What it does**: The backbone network architecture for feature extraction.

**Type**: String
**Default**: `"resnet50"`

**Available Options**:

| Backbone | Description | Best For |
|----------|-------------|----------|
| `resnet50` | ResNet-50 + FPN (original weights) | Standard use |
| `resnet50_v2` | ResNet-50 + FPN (improved V2 weights) | Better accuracy |

**Example**:
```toml
# Standard backbone
backbone = "resnet50"

# Improved backbone
backbone = "resnet50_v2"
```

---

#### `learning_rate`
**What it does**: Controls how much the model weights are updated during training.

**Type**: Float
**Default**: `5e-4`
**Recommended range**: `1e-5` to `1e-3`

**Example**:
```toml
# Default
learning_rate = 5e-4

# Fine-tuning (smaller updates)
learning_rate = 1e-4

# Faster learning (risk of instability)
learning_rate = 1e-3
```

---

#### `weight_decay`
**What it does**: L2 regularization to prevent overfitting.

**Type**: Float
**Default**: `5e-4`

**Example**:
```toml
weight_decay = 5e-4
```

---

#### `momentum`
**What it does**: Momentum for SGD optimizer to accelerate training.

**Type**: Float
**Default**: `0.9`

**Example**:
```toml
momentum = 0.9
```

---

#### `lr_scheduler_step_size`
**What it does**: Number of epochs between learning rate reductions.

**Type**: Integer
**Default**: `5`

**Example**:
```toml
# Reduce LR every 5 epochs
lr_scheduler_step_size = 5

# Reduce LR every 10 epochs
lr_scheduler_step_size = 10
```

---

#### `lr_scheduler_gamma`
**What it does**: Factor by which learning rate is reduced at each step.

**Type**: Float
**Default**: `0.1`

**Example**:
```toml
# Reduce LR to 10% (default)
lr_scheduler_gamma = 0.1

# Reduce LR to 50% (gentler)
lr_scheduler_gamma = 0.5
```

---

#### `transfer_learning`
**What it does**: Enable transfer learning from a previous checkpoint stored in the experiment.

**Type**: Boolean
**Default**: `False`

When enabled, the pipeline will attempt to load the `checkpoint-latest` artifact from the experiment and use it as the starting point for training. This is useful for:
- Continuing training from a previous run
- Fine-tuning on new data while preserving learned features
- Incremental learning scenarios

If the checkpoint is not available or has a backbone mismatch, the pipeline will fall back to ImageNet pretrained weights.

**Example**:
```toml
# Disable transfer learning (train from ImageNet weights)
transfer_learning = false

# Enable transfer learning (load from checkpoint-latest)
transfer_learning = true
```

---

### Export Parameters

#### `export_format`
**What it does**: Specifies the format for exporting the trained model.

**Type**: String
**Default**: `"torchscript"`

**Available Options**:

| Format | Description | Best For |
|--------|-------------|----------|
| `torchscript` | PyTorch TorchScript format | Production deployment, cross-platform |
| `onnx` | Open Neural Network Exchange | Interoperability, various runtimes |

**Example**:
```toml
[export_parameters]
# TorchScript (default)
export_format = "torchscript"

# ONNX
export_format = "onnx"
```

---

## Real-World Configuration Examples

### Example 1: Standard Production Training

**Goal**: Train a high-quality Mask R-CNN model for deployment.

```toml
[hyperparameters]
epochs = 100
batch_size = 4
image_size = 800
backbone = "resnet50_v2"
learning_rate = 5e-4
weight_decay = 5e-4
momentum = 0.9
lr_scheduler_step_size = 30
lr_scheduler_gamma = 0.1
transfer_learning = false

[export_parameters]
export_format = "torchscript"
```

**Why these settings**:
- V2 backbone for best accuracy
- 100 epochs for thorough training
- Standard image size for good mask quality
- TorchScript export for production deployment

**Expected training time**: 4-8 hours on single GPU (dataset dependent)

---

### Example 2: Fast Prototyping

**Goal**: Quickly test if the dataset works with Mask R-CNN.

```toml
[hyperparameters]
epochs = 10
batch_size = 2
image_size = 640
backbone = "resnet50"
learning_rate = 5e-4
```

**Why these settings**:
- Short training for fast feedback
- Small batch size to fit in memory
- Lower resolution for speed
- Can complete in 30-60 minutes

---

### Example 3: High Accuracy Priority

**Goal**: Maximum accuracy for precise segmentation.

```toml
[hyperparameters]
epochs = 150
batch_size = 2
image_size = 1024
backbone = "resnet50_v2"
learning_rate = 1e-4
weight_decay = 1e-3
lr_scheduler_step_size = 50
lr_scheduler_gamma = 0.1
transfer_learning = false
```

**Why these settings**:
- V2 backbone for best features
- Higher resolution for better masks
- Longer training for convergence
- Lower learning rate for stability
- Stronger regularization

---

### Example 5: Transfer Learning from Previous Run

**Goal**: Continue training from a previous checkpoint.

```toml
[hyperparameters]
epochs = 50
batch_size = 4
image_size = 800
backbone = "resnet50"
learning_rate = 1e-4
weight_decay = 5e-4
momentum = 0.9
lr_scheduler_step_size = 10
lr_scheduler_gamma = 0.1
transfer_learning = true
```

**Why these settings**:
- Transfer learning enabled to load previous checkpoint
- Lower learning rate for fine-tuning
- Same backbone as original training (required)
- Fewer epochs needed when starting from checkpoint

---

### Example 4: Limited GPU Memory

**Goal**: Train on 6-8 GB GPU.

```toml
[hyperparameters]
epochs = 50
batch_size = 1
image_size = 640
backbone = "resnet50"
learning_rate = 5e-4
```

**Why these settings**:
- Minimal batch size
- Lower resolution
- Standard backbone
- All to fit in limited VRAM

---

## Understanding Your Results

### Training Artifacts

After training completes, check Picsellia for:

- **Model weights**: Final checkpoint saved automatically
- **Training curves**: Loss components over epochs
- **Validation metrics**: Losses on validation set
- **Sample predictions**: Visualizations with masks

### Loss Components

Mask R-CNN reports multiple loss components:

| Loss | Description |
|------|-------------|
| `loss` | Total combined loss |
| `loss_classifier` | Classification loss for detected objects |
| `loss_box_reg` | Bounding box regression loss |
| `loss_mask` | Mask prediction loss |
| `loss_objectness` | RPN objectness loss |
| `loss_rpn_box_reg` | RPN box regression loss |

### Performance Metrics

Instance segmentation is evaluated using:
- **Mask mAP**: Mean Average Precision for masks
- **Box mAP**: Mean Average Precision for bounding boxes
- **mAP50**: AP at IoU threshold 0.5
- **mAP75**: AP at IoU threshold 0.75

---

## Troubleshooting Guide

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce `batch_size` (try 2, 1)
2. Reduce `image_size` (try 640, 512)
3. Reduce `workers` environment variable
4. Use gradient accumulation (requires code modification)

---

### Issue: Training Too Slow

**Solutions**:
1. Increase `batch_size` if memory allows
2. Increase `workers` for data loading
3. Reduce `image_size`
4. Ensure using GPU (not CPU)

---

### Issue: Model Not Learning

**Check**:
1. Is loss decreasing? Monitor training curves
2. Are annotations correct with valid segmentation masks?
3. Is learning rate appropriate? Try 1e-4
4. Is dataset large enough?

---

### Issue: Poor Mask Quality

**Consider**:
1. Increase `image_size` for finer details
2. Ensure training masks are high quality
3. Try more training epochs
4. Use `resnet50_v2` backbone
5. Lower `mask_thresh` during evaluation

---

### Issue: Loss is NaN or Exploding

**Check**:
1. Reduce `learning_rate` (try 1e-5)
2. Check for invalid annotations (zero-area masks)
3. Ensure images are not corrupted
4. Add gradient clipping (requires code modification)

---

## Best Practices

### 1. Choose the Right Backbone

```toml
# For standard accuracy
backbone = "resnet50"

# For best accuracy
backbone = "resnet50_v2"
```

### 2. Mask R-CNN vs Other Models

**Use Mask R-CNN when**:
- Need pixel-level segmentation masks
- Object instances need to be separated
- High mask quality is important
- Two-stage detection is acceptable

**Use YOLO Segmentation when**:
- Real-time inference is critical
- Mask precision is less important
- Single-stage detection preferred

### 3. Dataset Size Recommendations

Mask R-CNN performs best with:
- **Minimum**: 200 images per class
- **Recommended**: 500-2,000 images per class
- **Ideal**: 2,000+ images per class

### 4. Monitor Training Actively

Check every 10-20 epochs:
- Is total loss decreasing steadily?
- Is mask loss decreasing?
- Are validation losses improving?
- Any signs of overfitting?

### 5. Annotation Quality

For best results:
- Ensure masks precisely follow object boundaries
- Avoid overlapping masks for the same object
- Include objects at various scales
- Cover edge cases in your domain

---

## Getting Started Checklist

- [ ] Prepare annotated dataset in Picsellia (COCO format with segmentation)
- [ ] Verify annotations include polygon masks
- [ ] Choose backbone based on accuracy needs
- [ ] Set batch size based on GPU memory
- [ ] Start with 50 epochs for initial training
- [ ] Monitor training curves in Picsellia
- [ ] Evaluate on test set
- [ ] Export for deployment

---

## Related Pipelines

- **YOLOv7 Segmentation**: Alternative instance segmentation
- **YOLOv8 Training**: Object detection training
- **SAM-2 Fine-tuning**: Segment Anything Model
- **Dataset Tiler**: Process large images

---

## Support

**Need help?**
- Picsellia platform questions: Contact your Picsellia support team
- Mask R-CNN model questions: See TorchVision documentation
- Pipeline configuration help: Refer to this guide

---

## Key Advantages of Mask R-CNN

1. **High-quality masks**: Precise pixel-level segmentation
2. **Multi-task learning**: Boxes and masks learned jointly
3. **Well-established**: Mature architecture with extensive research
4. **Flexible backbone**: Support for different feature extractors
5. **Production-ready**: Widely used in industry applications

---

**Pipeline Version**: 1.0.0
**Type**: Training
**Framework**: PyTorch + TorchVision
**Last Updated**: 2026-01-23

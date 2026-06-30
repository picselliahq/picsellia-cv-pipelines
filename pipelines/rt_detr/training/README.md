# RT-DETR Training Pipeline

**Train state-of-the-art RT-DETR (Real-Time DEtection TRansformer) models for fast and accurate object detection.**

This Picsellia pipeline trains RT-DETR models using the Hugging Face Transformers library. RT-DETR is a real-time object detector that achieves excellent accuracy while maintaining high inference speed, making it ideal for production deployments.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A fully trained RT-DETR model fine-tuned on your data
- ✅ Model weights saved as experiment artifacts
- ✅ Evaluation metrics on your test set
- ✅ Training curves and visualizations in Picsellia
- ✅ Real-time performance optimized detector

## Quick Start Guide

### 🎯 Basic Training

**Goal**: Train RT-DETR with default settings.

```toml
[hyperparameters]
epochs = 50
batch_size = 8
image_size = 640
model_name = "PekingU/rtdetr_v2_r50vd"
```

This will fine-tune the RT-DETR ResNet-50 model on your dataset.

### 🎯 Fast Prototyping

**Goal**: Quick training for testing.

```toml
[hyperparameters]
epochs = 10
batch_size = 4
image_size = 640
model_name = "PekingU/rtdetr_v2_r50vd"
```

Shorter training for rapid iteration.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Parameters

#### `epochs`
**What it does**: Number of complete passes through the training dataset.

**Type**: Integer
**Default**: `50`
**Recommended range**: `10` to `100`

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
**Default**: `8`
**Recommended range**: `4` to `16`

**GPU Memory Guide**:
```
8 GB VRAM  → batch_size = 4-8
16 GB VRAM → batch_size = 8-12
24 GB VRAM → batch_size = 12-16
```

**Example**:
```toml
batch_size = 8
```

**Note**: RT-DETR models are memory-intensive compared to YOLO models. Use smaller batch sizes.

---

#### `image_size`
**What it does**: Size to which images are resized during training.

**Type**: Integer
**Default**: `640`
**Common values**: `480`, `640`, `800`

**Example**:
```toml
# Standard
image_size = 640

# Lower resolution for speed
image_size = 480

# Higher resolution for accuracy
image_size = 800
```

---

#### `model_name`
**What it does**: Hugging Face model repository ID to use as the base model.

**Type**: String
**Default**: `"PekingU/rtdetr_v2_r50vd"`

**Available Models**:

| Model | Backbone | Speed | Accuracy | Best For |
|-------|----------|-------|----------|----------|
| `PekingU/rtdetr_v2_r18vd` | ResNet-18 | Fastest | Good | Edge devices, real-time |
| `PekingU/rtdetr_v2_r34vd` | ResNet-34 | Fast | Better | Balanced applications |
| `PekingU/rtdetr_v2_r50vd` | ResNet-50 | Moderate | Best | Standard use (recommended) |
| `PekingU/rtdetr_v2_r101vd` | ResNet-101 | Slower | Excellent | High accuracy priority |

**Example**:
```toml
# Recommended default
model_name = "PekingU/rtdetr_v2_r50vd"

# Fastest model
model_name = "PekingU/rtdetr_v2_r18vd"

# Most accurate model
model_name = "PekingU/rtdetr_v2_r101vd"
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Standard Production Training

**Goal**: Train a high-quality RT-DETR model for deployment.

```toml
[hyperparameters]
epochs = 100
batch_size = 8
image_size = 640
model_name = "PekingU/rtdetr_v2_r50vd"
```

**Why these settings**:
- ResNet-50 backbone for best accuracy/speed balance
- 100 epochs for thorough training
- Standard image size

**Expected training time**: 3-6 hours on single GPU (dataset dependent)

---

### Example 2: Fast Prototyping

**Goal**: Quickly test if the dataset works with RT-DETR.

```toml
[hyperparameters]
epochs = 10
batch_size = 4
image_size = 640
model_name = "PekingU/rtdetr_v2_r50vd"
```

**Why these settings**:
- Short training for fast feedback
- Small batch size to fit in memory
- Can complete in 30-60 minutes

---

### Example 3: Edge Deployment

**Goal**: Train a fast model for edge devices or real-time applications.

```toml
[hyperparameters]
epochs = 80
batch_size = 8
image_size = 480
model_name = "PekingU/rtdetr_v2_r18vd"
```

**Why these settings**:
- Lightest backbone (ResNet-18)
- Lower resolution for faster inference
- Good for embedded systems

---

### Example 4: High Accuracy Priority

**Goal**: Maximum accuracy, inference speed is secondary.

```toml
[hyperparameters]
epochs = 150
batch_size = 4
image_size = 800
model_name = "PekingU/rtdetr_v2_r101vd"
```

**Why these settings**:
- Largest backbone (ResNet-101)
- Higher resolution for better detection
- Longer training for convergence
- Smaller batch due to memory constraints

---

### Example 5: Limited GPU Memory

**Goal**: Train on 6-8 GB GPU.

```toml
[hyperparameters]
epochs = 50
batch_size = 2
image_size = 480
model_name = "PekingU/rtdetr_v2_r18vd"
```

**Why these settings**:
- Small backbone
- Minimal batch size
- Lower resolution
- All to fit in limited VRAM

---

## 📊 Understanding Your Results

### Training Artifacts

After training completes, check Picsellia for:

- **Model weights**: Best checkpoint saved automatically
- **Training curves**: Loss over epochs
- **Validation metrics**: mAP scores
- **Sample predictions**: Visualizations

### Performance Metrics

RT-DETR is evaluated using:
- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Average across IoU thresholds 0.5 to 0.95
- **Inference speed**: FPS on your hardware

### Model Comparison

| Model Variant | Typical mAP50-95 | Inference Speed (GPU) |
|---------------|------------------|----------------------|
| R18 | ~45-50% | ~100 FPS |
| R34 | ~48-52% | ~80 FPS |
| R50 | ~50-55% | ~60 FPS |
| R101 | ~52-57% | ~40 FPS |

*Note: Actual results depend on your dataset quality and complexity.*

---

## ❓ Troubleshooting Guide

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce `batch_size` (try 4, 2, or even 1)
2. Reduce `image_size` (try 480 or 384)
3. Use smaller model (`rtdetr_v2_r18vd`)
4. Enable gradient checkpointing (requires code modification)

---

### Issue: Training Too Slow

**Solutions**:
1. Use smaller model variant (R18 instead of R101)
2. Reduce `image_size`
3. Ensure using GPU (not CPU)
4. Check GPU utilization (should be >80%)

---

### Issue: Model Not Learning

**Check**:
1. Is loss decreasing? Monitor training curves
2. Are annotations correct and complete?
3. Is dataset large enough? (RT-DETR needs more data than YOLO)
4. Try training for more epochs (50-100+)

---

### Issue: Poor Performance vs YOLO

**Consider**:
1. RT-DETR typically needs more training data than YOLO
2. Try training for more epochs
3. Ensure image quality is good
4. RT-DETR excels at real-time inference, not necessarily highest mAP

---

### Issue: Import/Model Loading Errors

**Check**:
1. Model name is correct (exactly as on Hugging Face)
2. Internet connection is working (downloads pretrained weights)
3. Sufficient disk space for model download

---

## 💡 Best Practices

### 1. Choose the Right Model Variant

```toml
# For production balance
model_name = "PekingU/rtdetr_v2_r50vd"

# For edge/mobile
model_name = "PekingU/rtdetr_v2_r18vd"

# For maximum accuracy
model_name = "PekingU/rtdetr_v2_r101vd"
```

### 2. RT-DETR vs YOLO Decision Guide

**Use RT-DETR when**:
- Real-time inference is critical
- Need consistent fast inference speeds
- Working with video streams
- Deploying to production at scale

**Use YOLO when**:
- Training data is limited (< 1000 images)
- Need highest possible accuracy
- Training time is constrained
- More mature ecosystem needed

### 3. Dataset Size Recommendations

RT-DETR performs best with:
- **Minimum**: 500 images per class
- **Recommended**: 1,000-5,000 images
- **Ideal**: 5,000+ images

For smaller datasets, consider YOLOv8 instead.

### 4. Monitor Training Actively

Check every 10-20 epochs:
- Is loss decreasing steadily?
- Are validation metrics improving?
- Any signs of overfitting?

### 5. Use Standard Image Size

```toml
image_size = 640  # Recommended default
```

Only change if you have specific requirements.

---

## 🚀 Getting Started Checklist

- [ ] Prepare annotated dataset in Picsellia (COCO format)
- [ ] Choose RT-DETR variant based on deployment needs
- [ ] Set batch size based on GPU memory
- [ ] Start with 50 epochs for initial training
- [ ] Monitor training curves in Picsellia
- [ ] Evaluate on test set
- [ ] Compare with baseline (e.g., YOLOv8)
- [ ] Export for deployment

---

## 🔗 Related Pipelines

- **YOLOv8 Training**: Alternative object detection training
- **YOLOv8 Pre-Annotation**: Generate training data
- **Dataset Tiler**: Process large images

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- RT-DETR model questions → See [Hugging Face RT-DETR](https://huggingface.co/PekingU)
- Pipeline configuration help → Refer to this guide

---

## 🎯 Key Advantages of RT-DETR

1. **Real-time performance**: Optimized for fast inference
2. **Transformer-based**: Modern architecture
3. **End-to-end training**: No NMS post-processing needed during inference
4. **Consistent speed**: Predictable inference time
5. **Production-ready**: Designed for deployment

---

**Pipeline Version**: 1.0.2
**Type**: Training
**Framework**: Hugging Face Transformers + RT-DETR
**Last Updated**: 2026-01-08

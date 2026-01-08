# YOLOv7 Segmentation Training Pipeline

**Train YOLOv7 models for instance segmentation tasks.**

This Picsellia pipeline trains YOLOv7 models for instance segmentation, detecting objects and generating pixel-perfect masks. YOLOv7 combines fast inference with accurate segmentation, making it ideal for real-time applications requiring both object detection and precise boundaries.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A fully trained YOLOv7 segmentation model
- ✅ Model weights saved as experiment artifacts
- ✅ Bounding boxes AND segmentation masks
- ✅ Evaluation metrics on your test set
- ✅ Fast inference-ready model

## Quick Start Guide

### 🎯 Basic Training

**Goal**: Train YOLOv7 segmentation with standard settings.

```toml
[hyperparameters]
epochs = 100
batch_size = 8
image_size = 640
device = "cuda:0"
train_set_split_ratio = 0.8
```

Standard configuration for instance segmentation.

### 🎯 Fast Prototyping

**Goal**: Quick training to validate dataset.

```toml
[hyperparameters]
epochs = 20
batch_size = 4
image_size = 640
device = "cuda:0"
train_set_split_ratio = 0.8
```

Shorter training for rapid iteration.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Parameters

YOLOv7 segmentation uses standard YOLO training parameters similar to YOLOv8.

**Key parameters**:
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Images per training batch (default: 8)
- `image_size`: Input image size (default: 640)
- `device`: Training hardware (default: "cuda:0")
- `train_set_split_ratio`: Train/validation split (default: 0.8)
- `seed`: Random seed for reproducibility (default: 0)
- `validate`: Run validation during training (default: true)

Plus all standard YOLO augmentation parameters (see YOLOv8 training docs).

---

## 🎓 Real-World Configuration Examples

### Example 1: Standard Segmentation

**Goal**: Train instance segmentation model.

```toml
[hyperparameters]
epochs = 150
batch_size = 8
image_size = 640
device = "cuda:0"
train_set_split_ratio = 0.8
seed = 42
validate = true
```

**Use case**: General purpose instance segmentation

---

### Example 2: High-Resolution Segmentation

**Goal**: Better accuracy for detailed masks.

```toml
[hyperparameters]
epochs = 100
batch_size = 4
image_size = 1280
device = "cuda:0"
train_set_split_ratio = 0.8
```

**Use case**: Medical imaging, high-detail requirements

---

### Example 3: Fast Inference Priority

**Goal**: Optimize for real-time performance.

```toml
[hyperparameters]
epochs = 100
batch_size = 16
image_size = 416
device = "cuda:0"
train_set_split_ratio = 0.8
```

**Use case**: Edge devices, real-time applications

---

## 📊 Understanding Your Results

### Instance Segmentation Output

YOLOv7 segmentation provides:
1. **Bounding boxes**: Object locations
2. **Class labels**: Object categories
3. **Segmentation masks**: Pixel-precise boundaries
4. **Confidence scores**: Detection confidence

### Evaluation Metrics

- **mAP (bbox)**: Object detection accuracy
- **mAP (mask)**: Segmentation mask accuracy
- **Precision/Recall**: For both boxes and masks
- **Inference speed**: FPS on your hardware

---

## ❓ Troubleshooting Guide

### Issue: Out of Memory

**Solutions**:
1. Reduce `batch_size` (try 4 or 2)
2. Reduce `image_size` (try 416)
3. Use smaller model variant

---

### Issue: Training Too Slow

**Solutions**:
1. Increase `batch_size` if memory allows
2. Reduce `image_size`
3. Use GPU (CPU is very slow)

---

### Issue: Poor Mask Quality

**Check**:
1. Are polygon annotations correct?
2. Train for more epochs
3. Use higher `image_size`
4. Check data quality

---

## 💡 Best Practices

### 1. Dataset Requirements

**Minimum**: 500 images with polygon masks
**Recommended**: 2,000+ images
**Annotations**: COCO format with polygon segmentation

### 2. Choose Image Size Wisely

```toml
image_size = 416   # Fast inference
image_size = 640   # Balanced (recommended)
image_size = 1280  # High accuracy
```

### 3. Use GPU

Segmentation training is intensive:
```toml
device = "cuda:0"
```

### 4. Monitor Both Metrics

Watch both bbox and mask mAP during training.

---

## 🚀 Getting Started Checklist

- [ ] Prepare dataset with polygon segmentation masks
- [ ] Set appropriate image_size
- [ ] Configure batch_size for your GPU
- [ ] Use GPU for training
- [ ] Monitor training curves
- [ ] Evaluate both bbox and mask performance
- [ ] Export for deployment

---

## 🔗 Related Pipelines

- **YOLOv8 Training**: Object detection (bbox only)
- **SAM2 Fine-Tuning**: Advanced segmentation
- **Dataset Tiler**: Process large images

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- YOLOv7 questions → See [YOLOv7 GitHub](https://github.com/WongKinYiu/yolov7)

---

**Pipeline Version**: 1.0.1
**Type**: Training
**Framework**: YOLOv7 Segmentation
**Last Updated**: 2026-01-08

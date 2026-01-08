# SAM 2 Fine-Tuning Pipeline

**Fine-tune Meta's Segment Anything Model 2 (SAM 2) for specialized segmentation tasks.**

This Picsellia pipeline fine-tunes SAM 2 models on your custom segmentation dataset. SAM 2 is Meta's advanced segmentation model that can segment any object with minimal prompting. Fine-tuning it on your specific domain improves accuracy for your use case while maintaining its powerful zero-shot capabilities.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A fine-tuned SAM 2 model specialized for your domain
- ✅ Model weights saved as experiment artifacts
- ✅ Improved segmentation accuracy on your data
- ✅ Evaluation metrics on your test set
- ✅ Maintained zero-shot segmentation capability

## Quick Start Guide

### 🎯 Basic Fine-Tuning

**Goal**: Fine-tune SAM 2 on your segmentation dataset.

```toml
[hyperparameters]
epochs = 10
batch_size = 4
learning_rate = 0.00001
device = "cuda:0"
```

Conservative settings for stable fine-tuning.

### 🎯 Fast Prototyping

**Goal**: Quick validation of fine-tuning approach.

```toml
[hyperparameters]
epochs = 3
batch_size = 2
learning_rate = 0.00001
device = "cuda:0"
```

Minimal training to test the pipeline.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Parameters

SAM 2 fine-tuning uses specialized parameters:

- **epochs**: Number of fine-tuning epochs (default: typically 5-20)
- **batch_size**: Images per batch (default: 2-4, SAM 2 is memory-intensive)
- **learning_rate**: Learning rate (default: 0.00001, very conservative)
- **device**: Training hardware (default: "cuda:0")
- **seed**: Random seed for reproducibility
- **train_set_split_ratio**: Train/validation split ratio

**Important**: SAM 2 requires significant GPU memory. Use small batch sizes.

---

## 🎓 Real-World Configuration Examples

### Example 1: Medical Image Segmentation

**Goal**: Fine-tune for medical imaging domain.

```toml
[hyperparameters]
epochs = 20
batch_size = 2
learning_rate = 0.000005
device = "cuda:0"
train_set_split_ratio = 0.8
```

**Why these settings**:
- More epochs for medical domain specialization
- Very small batch size (medical images are large)
- Very low learning rate for stability
- Preserve SAM 2's general capabilities

---

### Example 2: Satellite/Aerial Imagery

**Goal**: Adapt SAM 2 for geospatial segmentation.

```toml
[hyperparameters]
epochs = 15
batch_size = 4
learning_rate = 0.00001
device = "cuda:0"
train_set_split_ratio = 0.8
```

**Use case**: Building segmentation, land use classification, crop monitoring

---

### Example 3: Industrial Inspection

**Goal**: Fine-tune for defect segmentation.

```toml
[hyperparameters]
epochs = 10
batch_size = 4
learning_rate = 0.00001
device = "cuda:0"
train_set_split_ratio = 0.9
```

**Use case**: Manufacturing QC, surface defect detection

---

### Example 4: Limited GPU Memory (8-12 GB)

**Goal**: Fine-tune with hardware constraints.

```toml
[hyperparameters]
epochs = 10
batch_size = 1
learning_rate = 0.00001
device = "cuda:0"
```

**Why these settings**:
- Batch size of 1 minimizes memory usage
- May need to reduce image resolution as well

---

## 📊 Understanding Your Results

### SAM 2 Architecture

SAM 2 is designed for:
- **Promptable segmentation**: Segment with points, boxes, or masks
- **Zero-shot capability**: Segment novel objects
- **High accuracy**: State-of-the-art segmentation quality
- **Flexible prompting**: Multiple interaction modes

### Fine-Tuning Benefits

**Before fine-tuning**: Good general segmentation
**After fine-tuning**: Excellent domain-specific segmentation

**Improvements**:
- Better boundary accuracy for your object types
- Improved handling of domain-specific challenges
- Faster convergence with prompts
- Maintained zero-shot capability on new objects

### Evaluation Metrics

- **IoU (Intersection over Union)**: Mask overlap accuracy
- **Boundary F1**: Edge precision
- **Mean IoU**: Average across all classes
- **Per-class IoU**: Individual class performance

---

## ❓ Troubleshooting Guide

### Issue: Out of Memory (OOM)

**This is common with SAM 2. Solutions**:
1. Reduce `batch_size` to 1
2. Reduce image resolution
3. Use smaller SAM 2 variant
4. Use gradient checkpointing (if available)
5. Use GPU with more VRAM (24GB+ recommended)

```toml
batch_size = 1  # Minimum
```

---

### Issue: Training Unstable

**Solutions**:
1. Lower learning rate (try 0.000001)
2. Increase warmup steps
3. Use gradient clipping
4. Check data quality

```toml
learning_rate = 0.000005
```

---

### Issue: No Improvement Over Base Model

**Check**:
1. Is dataset large enough? (500+ images recommended)
2. Are annotations high quality?
3. Is learning rate too low?
4. Train for more epochs

---

### Issue: Catastrophic Forgetting

**Cause**: Model loses general capabilities during fine-tuning.

**Solutions**:
1. Use very low learning rate
2. Fine-tune fewer epochs
3. Use regularization techniques
4. Mix in general segmentation data

---

## 💡 Best Practices

### 1. Conservative Fine-Tuning

SAM 2 is already very powerful. Fine-tune gently:

```toml
epochs = 10           # Not too many
learning_rate = 0.00001  # Very low
batch_size = 2        # Small
```

### 2. Dataset Requirements

**Minimum**: 200 images with quality masks
**Recommended**: 1,000+ images
**Format**: COCO segmentation format with polygon masks

### 3. GPU Requirements

**Minimum**: 16 GB VRAM (with batch_size=1)
**Recommended**: 24 GB VRAM (batch_size=2-4)
**Ideal**: 40 GB+ VRAM (batch_size=4-8)

### 4. Evaluation Strategy

1. Test on held-out data from your domain
2. Test on general images (verify no catastrophic forgetting)
3. Compare against base SAM 2
4. Measure both IoU and boundary accuracy

### 5. When to Fine-Tune SAM 2

**Fine-tune when**:
- Have 500+ annotated images
- Domain is very specialized
- Need highest possible accuracy
- Have adequate GPU resources

**Use base SAM 2 when**:
- Limited annotations (< 200)
- General segmentation needs
- GPU memory constrained
- Quick deployment needed

---

## 🚀 Getting Started Checklist

- [ ] Have segmentation dataset (polygon masks)
- [ ] Verify GPU has adequate memory (16GB+ VRAM)
- [ ] Start with very low learning rate
- [ ] Use small batch size (1-4)
- [ ] Monitor training carefully
- [ ] Evaluate against base model
- [ ] Test on both domain and general images
- [ ] Export fine-tuned model

---

## 🔗 Related Pipelines

- **SAM3_Bbox**: Zero-shot detection with SAM
- **SAM3_Polygons**: Zero-shot segmentation
- **YOLOv7 Segmentation**: Alternative segmentation training

---

## 🎯 SAM 2 vs Other Segmentation Models

| Feature | SAM 2 | YOLOv8-Seg | Mask R-CNN |
|---------|-------|------------|------------|
| **Zero-shot** | ✅ Yes | ❌ No | ❌ No |
| **Promptable** | ✅ Yes | ❌ No | ❌ No |
| **Training data needed** | Low | High | High |
| **Inference speed** | Moderate | Fast | Slow |
| **Accuracy** | Excellent | Very Good | Good |
| **Memory usage** | High | Low | Moderate |

**Use SAM 2 when**:
- Need promptable segmentation
- Want zero-shot capability
- Have limited training data
- Quality over speed

**Use YOLOv8-Seg when**:
- Need real-time inference
- Have lots of training data
- Fixed class set
- Speed is critical

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- SAM 2 questions → See [SAM 2 GitHub](https://github.com/facebookresearch/segment-anything-2)

---

## ⚠️ Important Notes

1. **Memory intensive**: SAM 2 requires significant GPU memory
2. **Conservative fine-tuning**: Use low learning rates to preserve capabilities
3. **Quality over quantity**: Better to have fewer high-quality annotations
4. **Test thoroughly**: Verify no catastrophic forgetting on general images

---

**Pipeline Version**: 1.0.5
**Type**: Training (Fine-Tuning)
**Framework**: Meta SAM 2
**Task**: Instance Segmentation
**Last Updated**: 2026-01-08

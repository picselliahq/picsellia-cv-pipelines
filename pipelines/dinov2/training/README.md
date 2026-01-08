# DINOv2 Training Pipeline

**Train DINOv2 models for self-supervised visual feature learning and classification.**

This Picsellia pipeline fine-tunes Meta's DINOv2 (self-DIstillation with NO labels v2) models for image classification tasks. DINOv2 provides powerful visual representations learned through self-supervised learning, making it excellent for transfer learning and feature extraction.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A fine-tuned DINOv2 model for your classification task
- ✅ Model weights saved as experiment artifacts
- ✅ Evaluation metrics on your test set
- ✅ Powerful visual features for downstream tasks
- ✅ State-of-the-art image representations

## Quick Start Guide

### 🎯 Basic Training

**Goal**: Fine-tune DINOv2 for image classification.

```toml
[hyperparameters]
epochs = 20
batch_size = 32
learning_rate = 0.0001
device = "cuda:0"
```

Standard configuration for classification fine-tuning.

### 🎯 Fast Prototyping

**Goal**: Quick validation of dataset.

```toml
[hyperparameters]
epochs = 5
batch_size = 16
learning_rate = 0.0001
device = "cuda:0"
```

Shorter training for rapid testing.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Parameters

DINOv2 training uses standard classification parameters:

- **epochs**: Number of training epochs (default varies)
- **batch_size**: Images per batch (default varies)
- **learning_rate**: Learning rate for fine-tuning (default varies)
- **device**: Training hardware (default: "cuda:0")
- **seed**: Random seed for reproducibility
- **train_set_split_ratio**: Train/validation split ratio

**Note**: Specific defaults depend on the model variant and task.

---

## 🎓 Real-World Configuration Examples

### Example 1: Image Classification

**Goal**: Fine-tune for multi-class classification.

```toml
[hyperparameters]
epochs = 30
batch_size = 32
learning_rate = 0.0001
device = "cuda:0"
train_set_split_ratio = 0.8
```

**Use case**: Product categorization, scene classification

---

### Example 2: Few-Shot Learning

**Goal**: Train with limited data.

```toml
[hyperparameters]
epochs = 50
batch_size = 16
learning_rate = 0.00005
device = "cuda:0"
train_set_split_ratio = 0.9
```

**Why these settings**:
- More epochs to learn from limited data
- Lower learning rate for stability
- More training data (90/10 split)

---

### Example 3: Transfer Learning

**Goal**: Leverage pre-trained features for new domain.

```toml
[hyperparameters]
epochs = 20
batch_size = 32
learning_rate = 0.0001
device = "cuda:0"
```

**Use case**: Medical imaging, satellite imagery, specialized domains

---

## 📊 Understanding Your Results

### DINOv2 Capabilities

**Classification**: Assign images to categories
**Feature Extraction**: Generate powerful image embeddings
**Transfer Learning**: Adapt to new domains efficiently
**Few-Shot Learning**: Learn from limited examples

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1 Score**: Balanced metric
- **Confusion Matrix**: Class-wise predictions

---

## ❓ Troubleshooting Guide

### Issue: Out of Memory

**Solutions**:
1. Reduce `batch_size` (try 16 or 8)
2. Use smaller DINOv2 variant
3. Reduce image resolution

---

### Issue: Overfitting

**Solutions**:
1. Reduce learning rate
2. Add more training data
3. Use data augmentation
4. Increase regularization

---

### Issue: Poor Performance

**Check**:
1. Is learning rate appropriate?
2. Enough training epochs?
3. Data quality and labels correct?
4. Class balance issues?

---

## 💡 Best Practices

### 1. Dataset Requirements

**Minimum**: 100 images per class
**Recommended**: 500+ images per class
**Format**: Standard image classification dataset

### 2. Learning Rate Selection

```toml
# Small datasets
learning_rate = 0.00005

# Medium datasets
learning_rate = 0.0001

# Large datasets
learning_rate = 0.0002
```

### 3. Use GPU

DINOv2 models are large and need GPU:
```toml
device = "cuda:0"
```

### 4. Fine-Tuning Strategy

DINOv2 is pre-trained on massive datasets:
- Start with lower learning rates
- Fine-tune fewer epochs than training from scratch
- Leverage powerful pre-trained features

---

## 🚀 Getting Started Checklist

- [ ] Prepare image classification dataset
- [ ] Ensure balanced classes (if possible)
- [ ] Choose appropriate learning rate
- [ ] Set batch size for your GPU
- [ ] Use GPU for training
- [ ] Monitor training curves
- [ ] Evaluate on test set
- [ ] Export for deployment

---

## 🔗 Related Pipelines

- **CLIP Training**: Alternative vision-language approach
- **YOLOv8 Training**: For object detection tasks
- **Diversified Dataset Extractor**: Curate training data

---

## 🎯 DINOv2 Advantages

1. **Self-supervised pre-training**: Learned from billions of images
2. **Strong features**: Excellent visual representations
3. **Transfer learning**: Adapts well to new domains
4. **Few-shot capable**: Works with limited data
5. **State-of-the-art**: Competitive performance

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- DINOv2 questions → See [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)

---

**Pipeline Version**: 1.0.0
**Type**: Training
**Framework**: DINOv2 (Meta AI)
**Task**: Image Classification
**Last Updated**: 2026-01-08

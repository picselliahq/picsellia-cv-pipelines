# CLIP Training Pipeline

**Fine-tune OpenAI's CLIP (Contrastive Language-Image Pre-training) model for vision-language tasks.**

This Picsellia pipeline fine-tunes CLIP models using the Hugging Face Transformers library. CLIP learns to understand images through natural language descriptions, enabling zero-shot classification, image-text matching, and semantic search.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A fine-tuned CLIP model specialized for your domain
- ✅ Model weights saved as experiment artifacts
- ✅ Evaluation metrics on your test set
- ✅ A model that understands both images and text
- ✅ Capability for zero-shot classification

## Quick Start Guide

### 🎯 Basic Training

**Goal**: Fine-tune CLIP on your image-caption dataset.

```toml
[hyperparameters]
epochs = 10
batch_size = 8
learning_rate = 0.00005
model_name = "openai/clip-vit-base-patch32"
caption_prompt = "Describe the image"
```

This will fine-tune the base CLIP model on your data.

### 🎯 High-Resolution Training

**Goal**: Train with higher resolution for better quality.

```toml
[hyperparameters]
epochs = 20
batch_size = 4
learning_rate = 0.00005
model_name = "openai/clip-vit-large-patch14-336"
caption_prompt = "A photo of"
```

Using the large model with 336x336 resolution.

---

## 📋 Complete Parameter Reference

### 🔵 Essential Parameters

#### `epochs`
**What it does**: Number of complete passes through the training dataset.

**Type**: Integer
**Default**: `3`
**Recommended range**: `5` to `30`

**Example**:
```toml
# Quick fine-tuning
epochs = 5

# Standard training
epochs = 10

# Thorough training
epochs = 20
```

**Note**: CLIP fine-tuning typically needs fewer epochs than training from scratch.

---

#### `batch_size`
**What it does**: Number of image-text pairs processed together.

**Type**: Integer
**Default**: `8`
**Recommended range**: `4` to `32`

**GPU Memory Guide**:
```
Model: clip-vit-base-patch32
8 GB VRAM  → batch_size = 8-16
16 GB VRAM → batch_size = 16-32

Model: clip-vit-large-patch14-336
8 GB VRAM  → batch_size = 2-4
16 GB VRAM → batch_size = 4-8
24 GB VRAM → batch_size = 8-16
```

**Example**:
```toml
batch_size = 8
```

---

#### `learning_rate`
**What it does**: Step size for weight updates during training.

**Type**: Float
**Default**: `0.00005` (5e-5)
**Recommended range**: `0.00001` to `0.0001`

**How it works**:
- **Lower (1e-5)**: Slower, more stable training
- **Medium (5e-5)**: Balanced (recommended)
- **Higher (1e-4)**: Faster but may be unstable

**Example**:
```toml
# Conservative
learning_rate = 0.00001

# Standard (recommended)
learning_rate = 0.00005

# Aggressive
learning_rate = 0.0001
```

---

#### `warmup_steps`
**What it does**: Number of steps to gradually increase learning rate from zero.

**Type**: Integer
**Default**: `0`
**Recommended range**: `0` to `500`

**Why use warmup**: Prevents early training instability with large gradients.

**Example**:
```toml
# No warmup
warmup_steps = 0

# Standard warmup
warmup_steps = 100

# Long warmup for large datasets
warmup_steps = 500
```

---

#### `weight_decay`
**What it does**: L2 regularization strength to prevent overfitting.

**Type**: Float
**Default**: `0.1`
**Range**: `0.0` to `0.5`

**Example**:
```toml
# Standard regularization
weight_decay = 0.1

# Less regularization
weight_decay = 0.01

# No regularization
weight_decay = 0.0
```

---

#### `model_name`
**What it does**: Hugging Face model repository ID for the base CLIP model.

**Type**: String
**Default**: `"openai/clip-vit-large-patch14-336"`

**Available Models**:

| Model | Resolution | Params | Speed | Accuracy | Best For |
|-------|-----------|--------|-------|----------|----------|
| `openai/clip-vit-base-patch32` | 224×224 | 151M | Fastest | Good | Fast inference, limited resources |
| `openai/clip-vit-base-patch16` | 224×224 | 149M | Fast | Better | Balanced applications |
| `openai/clip-vit-large-patch14` | 224×224 | 428M | Moderate | Best | High accuracy |
| `openai/clip-vit-large-patch14-336` | 336×336 | 428M | Slower | Excellent | Maximum quality |

**Example**:
```toml
# Fastest model
model_name = "openai/clip-vit-base-patch32"

# Balanced
model_name = "openai/clip-vit-base-patch16"

# Best quality
model_name = "openai/clip-vit-large-patch14-336"
```

---

#### `caption_prompt`
**What it does**: Prompt template for generating or framing captions.

**Type**: String
**Default**: `"Describe the image"`

**How to use**: This prompt guides how text descriptions relate to images.

**Examples**:
```toml
# General description
caption_prompt = "Describe the image"

# Specific framing
caption_prompt = "A photo of"

# Question format
caption_prompt = "What is in this image?"

# Domain-specific
caption_prompt = "Medical image showing"
```

**Tips**:
- Keep it simple and consistent
- Match your dataset's caption style
- For labeled datasets: `"A photo of {label}"`
- For natural captions: `"Describe the image"`

---

## 🎓 Real-World Configuration Examples

### Example 1: Product Image Understanding

**Goal**: Fine-tune CLIP to understand product images and descriptions.

```toml
[hyperparameters]
epochs = 15
batch_size = 16
learning_rate = 0.00005
warmup_steps = 100
weight_decay = 0.1
model_name = "openai/clip-vit-base-patch16"
caption_prompt = "A product image showing"
```

**Why these settings**:
- Base model for speed
- Standard learning rate
- Moderate training duration
- Domain-specific prompt

---

### Example 2: Medical Imaging

**Goal**: Adapt CLIP for medical image-report matching.

```toml
[hyperparameters]
epochs = 20
batch_size = 8
learning_rate = 0.00003
warmup_steps = 200
weight_decay = 0.05
model_name = "openai/clip-vit-large-patch14-336"
caption_prompt = "Medical image showing"
```

**Why these settings**:
- Large high-res model for detail
- Conservative learning rate
- Longer warmup for stability
- Less regularization (medical data is precise)

---

### Example 3: Fashion/E-commerce

**Goal**: Enable image search with text queries.

```toml
[hyperparameters]
epochs = 10
batch_size = 32
learning_rate = 0.00005
warmup_steps = 100
weight_decay = 0.1
model_name = "openai/clip-vit-base-patch32"
caption_prompt = "A fashion item:"
```

**Why these settings**:
- Fast model for production
- Large batch for speed
- Standard settings
- Simple prompt

---

### Example 4: Zero-Shot Classification

**Goal**: Train CLIP for classifying images into text-described categories.

```toml
[hyperparameters]
epochs = 15
batch_size = 16
learning_rate = 0.00005
warmup_steps = 150
weight_decay = 0.1
model_name = "openai/clip-vit-large-patch14"
caption_prompt = "A photo of"
```

**Why these settings**:
- Large model for strong representations
- Standard resolution (224)
- Balanced training

**Use case**: After training, classify images with text like "a photo of a cat", "a photo of a dog"

---

### Example 5: Limited GPU Memory

**Goal**: Train on 6-8 GB GPU.

```toml
[hyperparameters]
epochs = 10
batch_size = 4
learning_rate = 0.00005
warmup_steps = 50
weight_decay = 0.1
model_name = "openai/clip-vit-base-patch32"
caption_prompt = "Describe the image"
```

**Why these settings**:
- Smallest model
- Minimal batch size
- Fits in limited VRAM

---

## 📊 Understanding Your Results

### Training Artifacts

After training completes:

- **Model weights**: Fine-tuned CLIP checkpoint
- **Training curves**: Contrastive loss over time
- **Embeddings**: Image and text representations
- **Evaluation metrics**: Image-text similarity scores

### CLIP Capabilities After Fine-Tuning

1. **Image-Text Matching**: Find images matching text queries
2. **Zero-Shot Classification**: Classify without labeled training data
3. **Semantic Search**: Search images with natural language
4. **Cross-Modal Retrieval**: Find text from images or images from text

### Performance Metrics

CLIP is evaluated on:
- **Contrastive Loss**: How well images match their captions
- **Accuracy**: Correct image-text pairs
- **Retrieval Metrics**: Image→Text and Text→Image recall

---

## ❓ Troubleshooting Guide

### Issue: Out of Memory

**Solutions**:
1. Reduce `batch_size` (try 4, 2, or 1)
2. Use smaller model (`clip-vit-base-patch32`)
3. Use 224 resolution instead of 336

---

### Issue: Training Unstable / Loss Explodes

**Solutions**:
1. Lower `learning_rate` (try 0.00001 or 0.00003)
2. Increase `warmup_steps` (try 200-500)
3. Reduce `batch_size`
4. Check data quality (captions must be meaningful)

---

### Issue: Poor Zero-Shot Performance

**Check**:
1. Are captions descriptive and diverse?
2. Is `caption_prompt` appropriate for your domain?
3. Try training longer (more epochs)
4. Use larger model variant

---

### Issue: Model Not Learning

**Check**:
1. Is contrastive loss decreasing?
2. Are image-caption pairs correct?
3. Dataset size sufficient? (CLIP needs 1000s of pairs)
4. Try higher learning rate (0.0001)

---

### Issue: Model Loading Errors

**Check**:
1. Model name exactly matches Hugging Face
2. Internet connection for downloading
3. Sufficient disk space

---

## 💡 Best Practices

### 1. Data Requirements

**Minimum**: 1,000 image-caption pairs
**Recommended**: 10,000+ pairs
**Ideal**: 100,000+ pairs

Each image needs a descriptive text caption.

### 2. Caption Quality Matters

**Good captions**:
- ✅ Descriptive: "A red sports car parked in front of a modern building"
- ✅ Specific: "Golden retriever playing with a tennis ball in the park"
- ✅ Consistent: Same style across dataset

**Poor captions**:
- ❌ Too short: "car"
- ❌ Too generic: "image"
- ❌ Inconsistent: Mix of styles

### 3. Choose Model by Use Case

```toml
# Speed priority
model_name = "openai/clip-vit-base-patch32"

# Balance
model_name = "openai/clip-vit-base-patch16"

# Quality priority
model_name = "openai/clip-vit-large-patch14-336"
```

### 4. Learning Rate Guidelines

```toml
# Small datasets (< 5K)
learning_rate = 0.00001

# Medium datasets (5K-50K)
learning_rate = 0.00005

# Large datasets (> 50K)
learning_rate = 0.0001
```

### 5. Use Warmup for Stability

```toml
# Always use warmup for fine-tuning
warmup_steps = 100  # or 5% of total steps
```

---

## 🚀 Getting Started Checklist

- [ ] Prepare dataset with image-caption pairs
- [ ] Format captions consistently
- [ ] Choose CLIP model variant
- [ ] Set appropriate learning rate
- [ ] Configure caption prompt for your domain
- [ ] Start with 10 epochs
- [ ] Monitor contrastive loss
- [ ] Evaluate on test set
- [ ] Test zero-shot capabilities

---

## 🔗 Use Cases After Training

### 1. Semantic Image Search
```python
# Search images with text
query = "red sports car"
# Returns images matching the description
```

### 2. Zero-Shot Classification
```python
# Classify without retraining
labels = ["cat", "dog", "bird"]
# Model predicts which label matches image
```

### 3. Image Captioning
```python
# Generate descriptions from images
# Or rank candidate captions
```

### 4. Content Moderation
```python
# Detect inappropriate content via text queries
query = "unsafe content"
```

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- CLIP model questions → See [OpenAI CLIP](https://github.com/openai/CLIP)
- Hugging Face docs → [CLIP on HF](https://huggingface.co/docs/transformers/model_doc/clip)

---

## 🎯 CLIP vs Other Models

| Task | Use CLIP | Use YOLO/RT-DETR |
|------|----------|------------------|
| Object detection | ❌ | ✅ |
| Zero-shot classification | ✅ | ❌ |
| Image-text matching | ✅ | ❌ |
| Semantic search | ✅ | ❌ |
| Bounding boxes | ❌ | ✅ |
| Natural language queries | ✅ | ❌ |

**CLIP excels at**: Understanding images through language, zero-shot tasks, cross-modal retrieval

**CLIP is not for**: Traditional object detection with bounding boxes

---

**Pipeline Version**: 1.0.6
**Type**: Training
**Framework**: Hugging Face Transformers + OpenAI CLIP
**Last Updated**: 2026-01-08

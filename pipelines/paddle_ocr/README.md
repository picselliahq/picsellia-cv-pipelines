# PaddleOCR Training Pipeline

**Train OCR models for text detection and recognition using PaddlePaddle.**

This Picsellia pipeline trains PaddleOCR models for Optical Character Recognition tasks. PaddleOCR provides both text detection (finding where text is) and text recognition (reading what the text says) in a two-stage approach, making it ideal for document processing, scene text recognition, and other OCR applications.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A trained text detection model (finds text regions)
- ✅ A trained text recognition model (reads the text)
- ✅ Both models saved as experiment artifacts
- ✅ Evaluation metrics on your test set
- ✅ Ready-to-deploy OCR system

## Quick Start Guide

### 🎯 Basic Training

**Goal**: Train both detection and recognition models with standard settings.

```toml
[hyperparameters]
bbox_epochs = 100
text_epochs = 100
bbox_batch_size = 8
text_batch_size = 8
bbox_learning_rate = 0.001
text_learning_rate = 0.001
device = "cuda:0"
max_text_length = 25
```

This will train both models for 100 epochs each.

### 🎯 Fast Prototyping

**Goal**: Quick training to validate your dataset.

```toml
[hyperparameters]
bbox_epochs = 20
text_epochs = 20
bbox_batch_size = 4
text_batch_size = 4
bbox_learning_rate = 0.001
text_learning_rate = 0.001
device = "cuda:0"
max_text_length = 25
```

Shorter training for rapid iteration.

---

## 📋 Complete Parameter Reference

### 🔵 Text Detection Parameters (bbox)

#### `bbox_epochs`
**What it does**: Number of epochs to train the text detection model.

**Type**: Integer
**Default**: `100`

**Example**:
```toml
bbox_epochs = 100
```

---

#### `bbox_batch_size`
**What it does**: Batch size for text detection training.

**Type**: Integer
**Default**: `8`

**Example**:
```toml
bbox_batch_size = 8
```

---

#### `bbox_learning_rate`
**What it does**: Learning rate for text detection model.

**Type**: Float
**Default**: `0.001`

**Example**:
```toml
bbox_learning_rate = 0.001
```

---

#### `bbox_save_epoch_step`
**What it does**: Save detection model checkpoint every N epochs.

**Type**: Integer
**Default**: `10`

**Example**:
```toml
bbox_save_epoch_step = 10
```

---

### 🔵 Text Recognition Parameters (text)

#### `text_epochs`
**What it does**: Number of epochs to train the text recognition model.

**Type**: Integer
**Default**: `100`

**Example**:
```toml
text_epochs = 100
```

---

#### `text_batch_size`
**What it does**: Batch size for text recognition training.

**Type**: Integer
**Default**: `8`

**Example**:
```toml
text_batch_size = 8
```

---

#### `text_learning_rate`
**What it does**: Learning rate for text recognition model.

**Type**: Float
**Default**: `0.001`

**Example**:
```toml
text_learning_rate = 0.001
```

---

#### `text_save_epoch_step`
**What it does**: Save recognition model checkpoint every N epochs.

**Type**: Integer
**Default**: `10`

**Example**:
```toml
text_save_epoch_step = 10
```

---

### 🔵 Shared Parameters

#### `device`
**What it does**: Hardware device for training.

**Type**: String
**Default**: `"cuda:0"`

**Example**:
```toml
device = "cuda:0"  # First GPU
device = "cpu"     # CPU (very slow)
```

---

#### `max_text_length`
**What it does**: Maximum length of text to recognize.

**Type**: Integer
**Default**: `25`

**Example**:
```toml
# Short text (license plates, labels)
max_text_length = 15

# Standard text (default)
max_text_length = 25

# Long text (paragraphs)
max_text_length = 50
```

---

#### `train_set_split_ratio`
**What it does**: Fraction of data for training vs validation.

**Type**: Float
**Default**: `0.8`

**Example**:
```toml
train_set_split_ratio = 0.8  # 80% train, 20% validation
```

---

#### `seed`
**What it does**: Random seed for reproducibility.

**Type**: Integer
**Default**: `0`

**Example**:
```toml
seed = 42
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Document OCR

**Goal**: Train models for document text extraction.

```toml
[hyperparameters]
bbox_epochs = 150
text_epochs = 150
bbox_batch_size = 8
text_batch_size = 8
bbox_learning_rate = 0.001
text_learning_rate = 0.001
device = "cuda:0"
max_text_length = 50
train_set_split_ratio = 0.8
```

**Why these settings**:
- Longer max_text_length for document paragraphs
- Standard epochs for good convergence
- Balanced batch sizes

---

### Example 2: License Plate Recognition

**Goal**: Fast OCR for short, standardized text.

```toml
[hyperparameters]
bbox_epochs = 100
text_epochs = 100
bbox_batch_size = 16
text_batch_size = 16
bbox_learning_rate = 0.001
text_learning_rate = 0.001
device = "cuda:0"
max_text_length = 10
train_set_split_ratio = 0.8
```

**Why these settings**:
- Short max_text_length (license plates are short)
- Larger batch sizes (simple, consistent text)
- Standard training

---

### Example 3: Scene Text (Signage, Posters)

**Goal**: Handle varied text in natural scenes.

```toml
[hyperparameters]
bbox_epochs = 200
text_epochs = 200
bbox_batch_size = 8
text_batch_size = 8
bbox_learning_rate = 0.0005
text_learning_rate = 0.0005
device = "cuda:0"
max_text_length = 30
train_set_split_ratio = 0.8
```

**Why these settings**:
- More epochs for challenging scene text
- Lower learning rates for stability
- Moderate max_text_length

---

## 📊 Understanding Your Results

### Two-Stage OCR Process

1. **Detection Model**: Finds bounding boxes around text regions
2. **Recognition Model**: Reads the text within each bounding box

Both models work together for complete OCR.

### Training Artifacts

After training:
- **Detection model weights**: Text localization
- **Recognition model weights**: Text reading
- **Training curves**: Separate for detection and recognition
- **Evaluation metrics**: Accuracy, precision, recall

---

## ❓ Troubleshooting Guide

### Issue: Out of Memory

**Solutions**:
1. Reduce batch sizes (try 4 or 2)
2. Reduce max_text_length
3. Use CPU (very slow but works)

---

### Issue: Recognition Accuracy Low

**Solutions**:
1. Increase text_epochs
2. Lower text_learning_rate
3. Check if max_text_length is appropriate
4. Verify text annotations are correct

---

### Issue: Detection Missing Text

**Solutions**:
1. Increase bbox_epochs
2. Check bounding box annotations
3. Verify images have clear text regions

---

### Issue: Training Very Slow

**Check**:
1. Using GPU? (CPU is 10-20x slower)
2. Batch sizes too small?
3. Dataset very large?

---

## 💡 Best Practices

### 1. Set Appropriate max_text_length

```toml
# Short text (5-10 chars)
max_text_length = 15

# Medium text (10-30 chars)
max_text_length = 30

# Long text (30+ chars)
max_text_length = 50
```

### 2. Balance Detection and Recognition Training

Usually train both for similar epochs:
```toml
bbox_epochs = 100
text_epochs = 100
```

### 3. GPU Highly Recommended

OCR training is intensive. Use GPU:
```toml
device = "cuda:0"
```

### 4. Dataset Requirements

**Minimum**: 500 images with text
**Recommended**: 2,000-5,000 images
**Annotations needed**: 
- Bounding boxes around text regions
- Transcribed text for each region

---

## 🚀 Getting Started Checklist

- [ ] Prepare dataset with text bounding boxes and transcriptions
- [ ] Set max_text_length based on your text
- [ ] Configure epochs based on dataset size
- [ ] Use GPU for training
- [ ] Monitor both detection and recognition metrics
- [ ] Evaluate on test set
- [ ] Deploy both models together

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- PaddleOCR questions → See [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)

---

**Pipeline Version**: 1.0.14-rc
**Type**: Training
**Framework**: PaddleOCR
**Last Updated**: 2026-01-08

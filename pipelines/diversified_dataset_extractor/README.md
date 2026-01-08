# Diversified Dataset Extractor Pipeline

**Select diverse, representative images from large datasets using embedding similarity.**

This Picsellia pipeline uses image embeddings to extract a diverse subset of images from a large dataset. By measuring visual similarity through neural network embeddings, it selects images that maximize diversity while maintaining representation of your data distribution.

## What You'll Get

After running this pipeline, you'll have:
- ✅ A new dataset with diverse, representative images
- ✅ Reduced dataset size while preserving variety
- ✅ Images selected based on visual similarity
- ✅ Efficient sampling from large datasets

## Quick Start Guide

### 🎯 Basic Diversification

**Goal**: Extract diverse subset from large dataset.

```toml
[parameters]
distance_threshold = 5
embedding_model = "openclip"
model_architecture = "ViT-B-16-plus-240"
pretrained_weights = "laion400m_e32"
```

This will select images that are visually different from each other.

### 🎯 More Diverse Selection

**Goal**: Maximize diversity (fewer similar images).

```toml
[parameters]
distance_threshold = 3
embedding_model = "openclip"
model_architecture = "ViT-B-16-plus-240"
pretrained_weights = "laion400m_e32"
```

Lower threshold = more strict = greater diversity.

### 🎯 Less Aggressive Filtering

**Goal**: Keep more images, less strict diversity.

```toml
[parameters]
distance_threshold = 8
embedding_model = "openclip"
model_architecture = "ViT-B-16-plus-240"
pretrained_weights = "laion400m_e32"
```

Higher threshold = more lenient = more images kept.

---

## 📋 Complete Parameter Reference

### 🔵 Diversity Control

#### `distance_threshold`
**What it does**: Minimum embedding distance between selected images.

**Type**: Integer
**Default**: `5`
**Recommended range**: `3` to `10`

**How it works**:
- Images are represented as embeddings (vectors)
- Distance measures how different two images are
- Only images farther apart than this threshold are kept

**Visual guide**:
```
threshold = 3   →  Very diverse (fewer images)
threshold = 5   →  Balanced (moderate diversity)
threshold = 8   →  Less strict (more images)
```

**When to adjust**:
- **Need maximum diversity?** → Lower to 3-4
- **Keep more images?** → Raise to 7-10
- **Balanced approach?** → Use default (5)

**Example**:
```toml
# Very diverse subset
distance_threshold = 3

# Balanced (default)
distance_threshold = 5

# Less aggressive filtering
distance_threshold = 8
```

---

### 🔵 Embedding Model Configuration

#### `embedding_model`
**What it does**: Type of embedding model to use.

**Type**: String
**Default**: `"openclip"`
**Options**: `"openclip"` (currently supported)

**Example**:
```toml
embedding_model = "openclip"
```

---

#### `model_architecture`
**What it does**: Specific model architecture for embeddings.

**Type**: String
**Default**: `"ViT-B-16-plus-240"`

**Available architectures**:
- `"ViT-B-16-plus-240"`: Vision Transformer Base (recommended)
- `"ViT-L-14"`: Vision Transformer Large (more accurate, slower)
- `"ViT-B-32"`: Vision Transformer Base 32 (faster, less accurate)

**Example**:
```toml
# Recommended balance
model_architecture = "ViT-B-16-plus-240"

# Higher quality embeddings
model_architecture = "ViT-L-14"

# Faster processing
model_architecture = "ViT-B-32"
```

---

#### `pretrained_weights`
**What it does**: Pre-trained weights for the embedding model.

**Type**: String
**Default**: `"laion400m_e32"`

**Common options**:
- `"laion400m_e32"`: Trained on LAION-400M (recommended)
- `"openai"`: Official OpenAI weights
- `"laion2b_s34b_b79k"`: Trained on LAION-2B (highest quality)

**Example**:
```toml
# Recommended default
pretrained_weights = "laion400m_e32"

# Highest quality
pretrained_weights = "laion2b_s34b_b79k"
```

---

## 🎓 Real-World Configuration Examples

### Example 1: Reduce Large Dataset by 50%

**Goal**: Select 5,000 diverse images from 10,000.

```toml
[parameters]
distance_threshold = 5
embedding_model = "openclip"
model_architecture = "ViT-B-16-plus-240"
pretrained_weights = "laion400m_e32"
```

**Result**: Approximately half the images, maximizing diversity.

---

### Example 2: Maximum Diversity Sampling

**Goal**: Get the most visually different images.

```toml
[parameters]
distance_threshold = 3
embedding_model = "openclip"
model_architecture = "ViT-L-14"
pretrained_weights = "laion2b_s34b_b79k"
```

**Why these settings**:
- Low threshold for strict diversity
- High-quality model for better similarity detection
- Best weights for accuracy

---

### Example 3: Fast Processing on Large Dataset

**Goal**: Quick diversification of 50,000+ images.

```toml
[parameters]
distance_threshold = 6
embedding_model = "openclip"
model_architecture = "ViT-B-32"
pretrained_weights = "laion400m_e32"
```

**Why these settings**:
- Faster model architecture
- Moderate threshold
- Standard weights

---

### Example 4: Curate Training Set

**Goal**: Select representative training data.

```toml
[parameters]
distance_threshold = 5
embedding_model = "openclip"
model_architecture = "ViT-B-16-plus-240"
pretrained_weights = "laion400m_e32"
```

**Use case**: From 20,000 unlabeled images, select 2,000 diverse ones for manual labeling.

---

## 📊 Understanding Your Results

### How It Works

1. **Embed images**: Each image → vector representation
2. **Measure similarity**: Calculate distances between all pairs
3. **Select diverse images**: Keep images farther than threshold
4. **Create new dataset**: Output contains selected images

### Output Dataset

- **Size**: Typically 30-70% of original (depends on threshold)
- **Content**: Visually diverse images
- **Annotations**: Preserved if present

### Selection Strategy

The algorithm:
- Starts with all images
- Compares each pair
- Removes similar images (distance < threshold)
- Keeps diverse representatives

---

## ❓ Troubleshooting Guide

### Issue: Too Few Images Selected

**Cause**: Threshold too low (too strict).

**Solution**:
```toml
distance_threshold = 7  # Increase
```

---

### Issue: Too Many Images (Not Diverse Enough)

**Cause**: Threshold too high (too lenient).

**Solution**:
```toml
distance_threshold = 3  # Decrease
```

---

### Issue: Processing Very Slow

**Solutions**:
1. Use faster model architecture:
   ```toml
   model_architecture = "ViT-B-32"
   ```
2. Process in smaller batches
3. Use more powerful hardware

---

### Issue: Results Not Intuitive

**Check**:
1. Try different model architecture
2. Verify images are loading correctly
3. Check embedding quality with sample comparisons

---

## 💡 Best Practices

### 1. Start with Default, Then Tune

```toml
# Start here
distance_threshold = 5

# If too few images → increase to 7-8
# If not diverse enough → decrease to 3-4
```

### 2. Use Case Guidance

```toml
# Labeling budget: Select diverse images to label
distance_threshold = 4

# Training efficiency: Remove redundant data
distance_threshold = 5

# Dataset curation: Sample from large collection
distance_threshold = 6
```

### 3. Model Selection

```toml
# Standard use (recommended)
model_architecture = "ViT-B-16-plus-240"

# Need best quality
model_architecture = "ViT-L-14"

# Need speed
model_architecture = "ViT-B-32"
```

### 4. Typical Workflow

1. **Start with large dataset**: e.g., 10,000 images
2. **Run diversification**: Extract ~3,000-5,000 diverse images
3. **Manual labeling**: Label the diverse subset
4. **Training**: Train model on curated data
5. **Iterate**: Repeat if needed

---

## 🚀 Getting Started Checklist

- [ ] Have large dataset in Picsellia
- [ ] Decide target diversity level (threshold)
- [ ] Choose embedding model architecture
- [ ] Run pipeline
- [ ] Review selected images
- [ ] Adjust threshold if needed
- [ ] Use diversified dataset for labeling/training

---

## 🔗 Related Pipelines

- **YOLOv8 Training**: Train on curated dataset
- **Grounding DINO**: Pre-annotate diverse images
- **Dataset Tiler**: Alternative preprocessing

---

## 🎯 Common Use Cases

### Active Learning
Select most informative images to label next.

### Dataset Reduction
Reduce storage/processing while maintaining coverage.

### Quality Curation
Remove near-duplicates and redundant images.

### Balanced Sampling
Ensure diverse representation before training.

---

## 📞 Support

**Need help?**
- Picsellia platform questions → Contact your Picsellia support team
- OpenCLIP questions → See [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)

---

**Pipeline Version**: 1.0.2
**Type**: Dataset Version Creation
**Supported Types**: All dataset types
**Last Updated**: 2026-01-08

# Training Pipelines User Guide

Complete guide to training computer vision models with Picsellia CV pipelines and the `pxl-pipeline` CLI.

---

## Table of Contents

- [Overview](#overview)
- [Object Detection Training](#object-detection-training)
- [Segmentation Training](#segmentation-training)
- [Foundation Model Training](#foundation-model-training)
- [OCR Training](#ocr-training)
- [Common Workflows](#common-workflows)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

Training pipelines fine-tune models on custom labeled datasets. All training pipelines:
- Support COCO format datasets
- Require train/val/test splits
- Export trained weights
- Generate evaluation metrics
- Integrate with Picsellia platform

### Prerequisites

- Labeled dataset in COCO format
- Train, validation, and test splits uploaded to Picsellia
- Pretrained model weights (optional but recommended)
- GPU with sufficient VRAM (8GB minimum, 16-24GB recommended)

---

## Object Detection Training

### YOLOv8 Training

**Purpose:** Train fast, accurate object detectors on custom datasets.

**Use Cases:**
- Real-time detection applications
- Edge deployment (mobile, embedded)
- Production systems requiring speed and accuracy
- Multi-class object detection

#### Quick Start

**1. Prepare dataset splits in Picsellia:**
- Train set (60-70% of data)
- Validation set (15-20% of data)
- Test set (15-20% of data)

**2. Create training configuration:**

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 100
batch_size = 16
image_size = 640
learning_rate = 0.001
patience = 20
save_period = 10
close_mosaic = 10
export_format = "onnx"

[input.dataset_collection.train]
id = "train-dataset-id"

[input.dataset_collection.val]
id = "val-dataset-id"

[input.dataset_collection.test]
id = "test-dataset-id"

[input.model_version]
id = "yolov8n-pretrained-weights"
```

**3. Test locally (quick validation):**

```bash
cd pipelines/yolov8/training
pxl-pipeline test training --run-config-file run_config.toml
```

**4. Deploy for full training:**

```bash
pxl-pipeline deploy training --organization my-org --bump minor
```

#### Parameter Guide

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 100 | Number of training epochs |
| `batch_size` | int | 16 | Batch size per device |
| `image_size` | int | 640 | Input image size (pixels) |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `patience` | int | 20 | Early stopping patience (epochs) |
| `save_period` | int | 10 | Checkpoint save frequency |
| `close_mosaic` | int | 10 | Epochs before disabling mosaic augmentation |
| `export_format` | string | "onnx" | Export format (onnx, torchscript, coreml) |
| `workers` | int | 8 | Data loading workers |

#### Model Variants

| Model | Parameters | Speed | Accuracy | VRAM | Use Case |
|-------|------------|-------|----------|------|----------|
| `yolov8n` | 3M | Fastest | Good | 4GB | Mobile, edge |
| `yolov8s` | 11M | Fast | Better | 6GB | General purpose |
| `yolov8m` | 26M | Medium | Very good | 8GB | Balanced |
| `yolov8l` | 44M | Slow | Excellent | 12GB | High accuracy |
| `yolov8x` | 68M | Slowest | Best | 16GB+ | Maximum accuracy |

#### Training Strategies

**Quick Iteration (Fast Training):**
```toml
[parameters]
epochs = 30
batch_size = 32
image_size = 512
learning_rate = 0.01
patience = 10
```

Use `pipelines/yolov8/fast_training/` for optimized quick training.

**Production Quality (Standard Training):**
```toml
[parameters]
epochs = 100
batch_size = 16
image_size = 640
learning_rate = 0.001
patience = 20
save_period = 10
```

**Maximum Accuracy:**
```toml
[parameters]
epochs = 300
batch_size = 8
image_size = 1024
learning_rate = 0.0005
patience = 50
save_period = 20
```

#### Example: Defect Detection

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "manufacturing-co"
env = "PROD"

[parameters]
epochs = 150
batch_size = 16
image_size = 640
learning_rate = 0.001
patience = 30
export_format = "onnx"

# Augmentation
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
degrees = 0.0  # No rotation for oriented defects
translate = 0.1
scale = 0.5
mosaic = 1.0

[input.dataset_collection.train]
id = "defects-train"

[input.dataset_collection.val]
id = "defects-val"

[input.dataset_collection.test]
id = "defects-test"

[input.model_version]
id = "yolov8m-pretrained"
```

---

### RT-DETR Training

**Purpose:** Train transformer-based detectors without NMS post-processing.

**Use Cases:**
- High-accuracy detection
- Research applications
- Scenarios requiring end-to-end differentiable detection

#### Quick Start

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 50
batch_size = 8
image_size = 640
model_name = "PekingU/rtdetr_v2_r50vd"
learning_rate = 5e-5
weight_decay = 0.05
warmup_ratio = 0.05

[input.dataset_collection.train]
id = "train-id"

[input.dataset_collection.val]
id = "val-id"

[input.dataset_collection.test]
id = "test-id"

[input.model_version]
id = "rtdetr-pretrained"
```

```bash
cd pipelines/rt_detr/training
pxl-pipeline test training --run-config-file run_config.toml
```

#### Model Variants

| Model | Backbone | Speed | Accuracy | VRAM |
|-------|----------|-------|----------|------|
| `rtdetr_v2_r18vd` | ResNet-18 | Fast | Good | 6GB |
| `rtdetr_v2_r34vd` | ResNet-34 | Medium | Better | 8GB |
| `rtdetr_v2_r50vd` | ResNet-50 | Slower | Very Good | 10GB |
| `rtdetr_v2_r101vd` | ResNet-101 | Slow | Excellent | 16GB |

#### Parameter Guide

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 50 | Training epochs |
| `batch_size` | int | 8 | Batch size |
| `image_size` | int | 640 | Input size |
| `model_name` | string | See above | HuggingFace model ID |
| `learning_rate` | float | 5e-5 | Learning rate |
| `weight_decay` | float | 0.05 | Weight decay |
| `warmup_ratio` | float | 0.05 | Warmup ratio |
| `conf_thresh` | float | 0.25 | Inference confidence threshold |

#### Performance Tips

**For Better Accuracy:**
```toml
epochs = 100
learning_rate = 1e-5  # Lower LR
model_name = "PekingU/rtdetr_v2_r101vd"  # Larger backbone
```

**For Faster Training:**
```toml
batch_size = 16
model_name = "PekingU/rtdetr_v2_r18vd"  # Smaller backbone
```

---

## Segmentation Training

### YOLOv7 Segmentation

**Purpose:** Train instance segmentation models (masks + bounding boxes).

**Use Cases:**
- Instance segmentation tasks
- Precise object boundaries needed
- Multi-object scenes

#### Quick Start

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 100
batch_size = 8
image_size = 640
learning_rate = 0.001
patience = 20

[input.dataset_collection.train]
id = "segmentation-train"

[input.dataset_collection.val]
id = "segmentation-val"

[input.dataset_collection.test]
id = "segmentation-test"

[input.model_version]
id = "yolov7-seg-pretrained"
```

```bash
cd pipelines/yolov7_segmentation
pxl-pipeline test yolov7_segmentation --run-config-file run_config.toml
```

**Note:** Dataset must contain polygon/segmentation annotations.

---

### SAM-2 Fine-tuning

**Purpose:** Fine-tune Segment Anything Model 2 for specific domains.

**Use Cases:**
- Medical imaging segmentation
- Domain-specific segmentation
- High-quality mask generation

#### Quick Start

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 30
batch_size = 4
learning_rate = 1e-5
freeze_backbone = true

[input.dataset_collection.train]
id = "medical-train"

[input.dataset_collection.val]
id = "medical-val"

[input.model_version]
id = "sam2-base"
```

```bash
cd pipelines/sam_2
pxl-pipeline test sam_2 --run-config-file run_config.toml
```

---

## Foundation Model Training

### CLIP Training

**Purpose:** Train vision-language models for image-text matching.

**Use Cases:**
- Image classification via text
- Zero-shot classification
- Image retrieval
- Multimodal embeddings

#### Quick Start

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 50
batch_size = 32
learning_rate = 1e-4
image_size = 224
text_max_length = 77

[input.dataset_collection.train]
id = "image-text-pairs-train"

[input.dataset_collection.val]
id = "image-text-pairs-val"

[input.model_version]
id = "clip-vit-base"
```

```bash
cd pipelines/clip
pxl-pipeline test clip --run-config-file run_config.toml
```

**Dataset Requirements:**
- Images with associated text descriptions
- Captions or class labels

---

### DINOv2 Training

**Purpose:** Self-supervised vision transformer training.

**Use Cases:**
- Feature extraction
- Pretraining on unlabeled data
- Transfer learning base models

#### Quick Start

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 100
batch_size = 64
learning_rate = 1e-4
image_size = 224

[input.dataset_collection.train]
id = "unlabeled-images"

[input.model_version]
id = "dinov2-base"
```

```bash
cd pipelines/dinov2
pxl-pipeline test dinov2 --run-config-file run_config.toml
```

**Note:** Does not require labeled data.

---

## OCR Training

### Paddle OCR

**Purpose:** Train text detection and recognition models.

**Use Cases:**
- Document OCR
- Scene text recognition
- Multi-language text
- Custom fonts/styles

#### Quick Start

```toml
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 100
batch_size = 32
learning_rate = 0.001
detection_model = "ch_PP-OCRv3_det"
recognition_model = "ch_PP-OCRv3_rec"

[input.dataset_collection.train]
id = "ocr-train"

[input.dataset_collection.val]
id = "ocr-val"

[input.model_version]
id = "paddleocr-pretrained"
```

```bash
cd pipelines/paddle_ocr
pxl-pipeline test paddle_ocr --run-config-file run_config.toml
```

**Dataset Requirements:**
- Images with text bounding boxes
- Transcriptions for recognition training

---

## Common Workflows

### Workflow 1: Train Custom Detector from Scratch

```bash
# 1. Prepare datasets (train/val/test splits)
# 2. Upload to Picsellia

# 3. Create training config
cat > yolo_config.toml << EOF
[job]
type = "TRAINING"
[parameters]
epochs = 100
batch_size = 16
image_size = 640
[input.dataset_collection.train]
id = "my-train-id"
[input.dataset_collection.val]
id = "my-val-id"
[input.dataset_collection.test]
id = "my-test-id"
[input.model_version]
id = "yolov8n-pretrained"
EOF

# 4. Quick test (5-10 epochs locally)
pxl-pipeline test training --run-config-file yolo_config.toml

# 5. Deploy for full training
pxl-pipeline deploy training --organization my-org --bump minor

# 6. Monitor training in Picsellia UI
# 7. Download trained weights
# 8. Run inference or deploy
```

### Workflow 2: Fine-tune Pretrained Model

```bash
# 1. Start with similar pretrained model
# 2. Use lower learning rate

cat > finetune_config.toml << EOF
[job]
type = "TRAINING"
[parameters]
epochs = 50
batch_size = 8
learning_rate = 0.0001  # Lower LR
freeze_backbone = true   # Freeze early layers
[input.model_version]
id = "my-pretrained-model"
EOF

# 3. Train
pxl-pipeline test training --run-config-file finetune_config.toml
```

### Workflow 3: Hyperparameter Tuning

```bash
# Test different configurations

# Config 1: Baseline
cat > config1.toml << EOF
[parameters]
learning_rate = 0.001
batch_size = 16
EOF

pxl-pipeline test training --run-config-file config1.toml

# Config 2: Higher LR
cat > config2.toml << EOF
[parameters]
learning_rate = 0.01
batch_size = 16
EOF

pxl-pipeline test training --run-config-file config2.toml

# Config 3: Larger batch
cat > config3.toml << EOF
[parameters]
learning_rate = 0.001
batch_size = 32
EOF

pxl-pipeline test training --run-config-file config3.toml

# Compare results in Picsellia UI
# Select best configuration
# Deploy with full epochs
```

---

## Hyperparameter Tuning

### Learning Rate

**Start:** `0.001`

**Too high:** Loss diverges, training unstable
**Too low:** Training too slow, may not converge

**Tuning strategy:**
```toml
# Try exponential scale
learning_rate = 0.0001  # Conservative
learning_rate = 0.001   # Standard
learning_rate = 0.01    # Aggressive
```

**With learning rate scheduling:**
```toml
learning_rate = 0.01
warmup_ratio = 0.05      # Warm up first 5%
lr_decay = "cosine"      # Cosine annealing
```

### Batch Size

**Considerations:**
- VRAM limited
- Training stability
- Convergence speed

**Guidelines:**

| VRAM | YOLOv8n | YOLOv8s | YOLOv8m | YOLOv8l |
|------|---------|---------|---------|---------|
| 8GB  | 32      | 16      | 8       | 4       |
| 12GB | 64      | 32      | 16      | 8       |
| 16GB | 96      | 48      | 24      | 12      |
| 24GB | 128     | 64      | 32      | 16      |

**Effective batch size:**
```toml
batch_size = 8
accumulation_steps = 4  # Effective batch size = 32
```

### Image Size

**Common sizes:** 416, 512, 640, 800, 1024

**Trade-offs:**
- Smaller: Faster training, less detail
- Larger: Slower training, more detail

**Guidelines:**
```toml
image_size = 416   # Fast, mobile deployment
image_size = 640   # Standard, balanced
image_size = 1024  # Slow, maximum accuracy
```

### Epochs and Early Stopping

```toml
epochs = 100      # Maximum epochs
patience = 20     # Stop if no improvement for 20 epochs
```

**Guidelines:**
- Small dataset (< 1000 images): 50-100 epochs
- Medium dataset (1000-10000): 100-200 epochs
- Large dataset (> 10000): 200-300 epochs

### Augmentation

**Light augmentation:**
```toml
hsv_h = 0.015
hsv_s = 0.4
hsv_v = 0.4
degrees = 10
translate = 0.1
scale = 0.2
mosaic = 1.0
```

**Heavy augmentation:**
```toml
hsv_h = 0.03
hsv_s = 0.7
hsv_v = 0.7
degrees = 20
translate = 0.2
scale = 0.5
mosaic = 1.0
mixup = 0.5
```

**Disable for final epochs:**
```toml
close_mosaic = 10  # Disable mosaic last 10 epochs
```

---

## Troubleshooting

### Issue: Training Loss Not Decreasing

**Possible Causes:**
1. Learning rate too low
2. Model too simple for task
3. Data quality issues
4. Incorrect labels

**Solutions:**
```toml
# Try higher learning rate
learning_rate = 0.01

# Use larger model
model_name = "yolov8m"  # was yolov8n

# Verify data quality
# Check annotations in Picsellia UI
```

### Issue: Loss Exploding

**Possible Causes:**
1. Learning rate too high
2. Bad initialization
3. Gradient instability

**Solutions:**
```toml
# Lower learning rate
learning_rate = 0.0001

# Add gradient clipping
grad_clip = 1.0

# Add warmup
warmup_ratio = 0.1
```

### Issue: Overfitting

**Symptoms:**
- Training loss low
- Validation loss high
- Large gap between train/val

**Solutions:**
```toml
# Increase regularization
weight_decay = 0.0005

# More augmentation
hsv_s = 0.7
mixup = 0.5

# Early stopping
patience = 15

# More training data
# Add dropout (model-dependent)
```

### Issue: Underfitting

**Symptoms:**
- Both train and val loss high
- Model not learning

**Solutions:**
```toml
# Use larger model
model_name = "yolov8l"

# Train longer
epochs = 200

# Higher learning rate
learning_rate = 0.001

# Less regularization
weight_decay = 0.0001
```

### Issue: Out of Memory (OOM)

**Solutions:**
```toml
# Reduce batch size
batch_size = 4

# Smaller image size
image_size = 512

# Use smaller model
model_name = "yolov8n"

# Reduce workers
workers = 2
```

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Issue: Training Too Slow

**Solutions:**
```toml
# Increase batch size
batch_size = 32

# Increase workers
workers = 16

# Mixed precision training (usually automatic)
fp16 = true

# Smaller image size
image_size = 512
```

**System-level:**
```bash
# Use NVMe SSD for datasets
# Ensure CUDA is properly installed
# Check GPU utilization with nvidia-smi
```

### Issue: Poor Test Performance

**Possible Causes:**
1. Overfitting
2. Train/test distribution mismatch
3. Insufficient training

**Solutions:**
```toml
# More training data
# Better augmentation
# Longer training
epochs = 200

# Ensemble models
# Test-time augmentation
```

---

## Best Practices

### Data Preparation
1. **Balance classes** - Avoid extreme class imbalance
2. **Quality over quantity** - Clean labels more important than size
3. **Diverse data** - Cover all scenarios
4. **Proper splits** - 60/20/20 or 70/15/15 train/val/test
5. **Consistent annotation** - Use same labeling guidelines

### Training Strategy
1. **Start small** - Test with few epochs locally
2. **Use pretrained weights** - Transfer learning almost always helps
3. **Monitor metrics** - Watch train and val loss
4. **Save checkpoints** - Regular checkpointing
5. **Document experiments** - Track what you tried

### Hyperparameter Selection
1. **Start with defaults** - Then tune incrementally
2. **One at a time** - Change one parameter per experiment
3. **Log everything** - Use Picsellia experiment tracking
4. **Grid/random search** - For systematic tuning
5. **Early stopping** - Prevent overtraining

### Deployment
1. **Test thoroughly** - Validate on held-out data
2. **Export optimized** - ONNX for inference
3. **Benchmark speed** - Measure FPS before deployment
4. **Version models** - Track model versions
5. **Monitor production** - Track real-world performance

---

## Performance Benchmarks

### YOLOv8 Training Time (Single GPU)

| Model | Dataset Size | Epochs | RTX 3090 | A100 |
|-------|--------------|--------|----------|------|
| YOLOv8n | 1000 images | 100 | 1 hour | 30 min |
| YOLOv8s | 1000 images | 100 | 2 hours | 1 hour |
| YOLOv8m | 1000 images | 100 | 4 hours | 2 hours |
| YOLOv8l | 1000 images | 100 | 6 hours | 3 hours |

**Factors affecting speed:**
- Image resolution
- Number of classes
- Augmentation complexity
- Batch size
- Hardware

---

## Next Steps

- **Deploy trained models:** Use for inference in Picsellia
- **Process datasets:** See [PROCESSING_PIPELINES_GUIDE.md](PROCESSING_PIPELINES_GUIDE.md)
- **Create custom pipelines:** See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **API integration:** Use Picsellia Python SDK

---

**Need Help?**
- [Picsellia CV Engine Docs](https://picselliahq.github.io/picsellia-cv-engine)
- [GitHub Issues](https://github.com/picselliahq/picsellia-cv-pipelines/issues)
- Email: support@picsellia.com

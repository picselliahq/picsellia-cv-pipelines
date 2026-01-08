# Documentation

Welcome to the Picsellia CV Pipelines documentation! This repository contains 17 production-ready computer vision pipelines managed through the `pxl-pipeline` CLI.

---

## 📚 Main Guides

### [Main README](../README.md)
**Start here!** Complete overview with quick start guide, installation, CLI basics, and usage examples.

### [Processing Pipelines Guide](PROCESSING_PIPELINES_GUIDE.md)
Complete guide to pre-annotation and dataset processing pipelines.

### [Training Pipelines Guide](TRAINING_PIPELINES_GUIDE.md)
Complete guide to training custom computer vision models.

### [Developer Guide](DEVELOPER_GUIDE.md)
Complete guide for creating custom pipelines from scratch.

---

## 🚀 Quick Start

```bash
# Install
pip install picsellia-cv-engine picsellia-pipelines-cli

# Authenticate
pxl-pipeline login --organization my-org --env PROD --token $TOKEN

# Test a pipeline
cd pipelines/SAM3_Bbox
pxl-pipeline test SAM3_Bbox --run-config-file run_config.toml

# Deploy
pxl-pipeline deploy SAM3_Bbox --organization my-org --bump patch
```

---

## 📖 Pipeline-Specific Documentation

Each pipeline has its own comprehensive README with parameters, examples, and best practices.

### Pre-Annotation Pipelines

Automatically generate annotations using foundation models.

| Pipeline | Description | README |
|----------|-------------|--------|
| **SAM-3 Bbox** | Zero-shot object detection with bounding boxes | [📄 README](../pipelines/SAM3_Bbox/README.md) |
| **SAM-3 Polygons** | Zero-shot segmentation with polygon masks | [📄 README](../pipelines/SAM3_polygons/README.md) |
| **Grounding DINO** | Text-guided zero-shot detection | [📄 README](../pipelines/grounding_dino/README.md) |
| **YOLOv8 Pre-annotation** | Fast detection with pretrained COCO weights | [📄 README](../pipelines/yolov8/pre_annotation/README.md) |

### Training Pipelines

Fine-tune models on your custom labeled datasets.

#### Object Detection

| Pipeline | Description | README |
|----------|-------------|--------|
| **YOLOv8 Training** | Train YOLOv8 detectors (full guide) | [📄 README](../pipelines/yolov8/training/README.md) |
| **YOLOv8 Fast Training** | Quick iteration training mode | [📄 README](../pipelines/yolov8/fast_training/README.md) |
| **RT-DETR** | Real-Time Detection Transformer | [📄 README](../pipelines/rt_detr/training/README.md) |

#### Segmentation

| Pipeline | Description | README |
|----------|-------------|--------|
| **YOLOv7 Segmentation** | Instance segmentation training | [📄 README](../pipelines/yolov7_segmentation/README.md) |
| **SAM-2 Fine-tuning** | Fine-tune Segment Anything Model 2 | [📄 README](../pipelines/sam_2/fine_tuning/README.md) |

#### Foundation Models

| Pipeline | Description | README |
|----------|-------------|--------|
| **CLIP** | Vision-language model training | [📄 README](../pipelines/clip/training/README.md) |
| **DINOv2** | Self-supervised vision transformer | [📄 README](../pipelines/dinov2/training/README.md) |

#### Specialized

| Pipeline | Description | README |
|----------|-------------|--------|
| **Paddle OCR** | Text detection and recognition training | [📄 README](../pipelines/paddle_ocr/README.md) |

### Dataset Processing Pipelines

Transform, augment, and prepare your datasets.

| Pipeline | Description | README |
|----------|-------------|--------|
| **Dataset Tiler** | Split large images into tiles with overlap | [📄 README](../pipelines/dataset_tiler/README.md) |
| **Albumentations** | Image augmentation with 40+ transformations | [📄 README](../pipelines/albumentations_processing/README.md) |
| **Bounding Box Cropper** | Extract crops around annotations | [📄 README](../pipelines/bounding_box_cropper/README.md) |
| **Diversified Extractor** | Sample diverse images using embeddings | [📄 README](../pipelines/diversified_dataset_extractor/README.md) |

---

## 🎯 Quick Navigation by Use Case

### I want to...

**...pre-annotate my dataset**
- Multi-class detection → [SAM-3 Bbox](../pipelines/SAM3_Bbox/README.md)
- Precise segmentation → [SAM-3 Polygons](../pipelines/SAM3_polygons/README.md)
- Text-guided detection → [Grounding DINO](../pipelines/grounding_dino/README.md)
- Fast COCO detection → [YOLOv8 Pre-annotation](../pipelines/yolov8/pre_annotation/README.md)

**...train an object detector**
- Standard training → [YOLOv8 Training](../pipelines/yolov8/training/README.md)
- Quick testing → [YOLOv8 Fast Training](../pipelines/yolov8/fast_training/README.md)
- Transformer-based → [RT-DETR](../pipelines/rt_detr/training/README.md)

**...train a segmentation model**
- Instance segmentation → [YOLOv7 Segmentation](../pipelines/yolov7_segmentation/README.md)
- Fine-tune SAM-2 → [SAM-2 Fine-tuning](../pipelines/sam_2/fine_tuning/README.md)

**...process large images**
- Tile images → [Dataset Tiler](../pipelines/dataset_tiler/README.md)

**...augment my dataset**
- Image augmentation → [Albumentations](../pipelines/albumentations_processing/README.md)

**...extract object crops**
- Crop around boxes → [Bounding Box Cropper](../pipelines/bounding_box_cropper/README.md)

**...train specialized models**
- Vision-language → [CLIP](../pipelines/clip/training/README.md)
- Self-supervised → [DINOv2](../pipelines/dinov2/training/README.md)
- OCR → [Paddle OCR](../pipelines/paddle_ocr/README.md)

**...create a custom pipeline**
- Developer guide → [Developer Guide](DEVELOPER_GUIDE.md)

---

## 🔧 pxl-pipeline CLI Reference

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `login` | Authenticate with Picsellia | `pxl-pipeline login --organization my-org --token $TOKEN` |
| `init` | Create new pipeline from template | `pxl-pipeline init my_pipeline --type processing --template pre_annotation` |
| `test` | Test pipeline locally | `pxl-pipeline test <pipeline> --run-config-file config.toml` |
| `deploy` | Deploy to Picsellia cloud | `pxl-pipeline deploy <pipeline> --organization my-org --bump patch` |

**Full CLI documentation:** [Picsellia CV Engine - CLI Usage](https://picselliahq.github.io/picsellia-cv-engine/usage/)

---

## 📖 What Each README Contains

Every pipeline README includes:

- ✅ **Title and Purpose** - What the pipeline does
- ✅ **What You'll Get** - Expected outputs
- ✅ **Quick Start Guide** - Copy-paste ready examples
- ✅ **Complete Parameter Reference** - Every parameter explained with:
  - Types, defaults, and valid ranges
  - When and how to adjust
  - Visual guides and decision matrices
- ✅ **Real-World Examples** - 3-6 practical configurations
- ✅ **Understanding Results** - How to interpret outputs
- ✅ **Troubleshooting** - Common issues and solutions
- ✅ **Best Practices** - Expert tips
- ✅ **Getting Started Checklist** - Step-by-step onboarding

---

## 🎓 Learning Path

### For First-Time Users

1. **Start here** → [Main README](../README.md)
2. **Install and authenticate** → [Quick Start](#quick-start)
3. **Pick a use case:**
   - Processing data? → [Processing Guide](PROCESSING_PIPELINES_GUIDE.md)
   - Training models? → [Training Guide](TRAINING_PIPELINES_GUIDE.md)
4. **Read pipeline README** → Choose from tables above
5. **Test locally** → Follow pipeline's quick start
6. **Deploy to Picsellia** → Use `pxl-pipeline deploy`

### For Developers

1. **Environment setup** → [Developer Guide - Setup](DEVELOPER_GUIDE.md#getting-started)
2. **Understand architecture** → [Developer Guide - Architecture](DEVELOPER_GUIDE.md#pipeline-architecture)
3. **Create pipeline** → [Developer Guide - Creating](DEVELOPER_GUIDE.md#creating-a-new-pipeline)
4. **Study examples** → Read existing pipeline code
5. **Test and deploy** → [Developer Guide - Testing](DEVELOPER_GUIDE.md#testing-locally)

---

## 🆘 Getting Help

### Documentation

- **Processing issues?** → [Processing Guide - Troubleshooting](PROCESSING_PIPELINES_GUIDE.md#troubleshooting)
- **Training issues?** → [Training Guide - Troubleshooting](TRAINING_PIPELINES_GUIDE.md#troubleshooting)
- **Development issues?** → [Developer Guide - Troubleshooting](DEVELOPER_GUIDE.md#troubleshooting-development)
- **Pipeline-specific?** → Check pipeline's README

### Support Channels

**GitHub Issues**
- Bug reports: [Report a bug](https://github.com/picselliahq/picsellia-cv-pipelines/issues/new)
- Feature requests: [Request a feature](https://github.com/picselliahq/picsellia-cv-pipelines/issues/new)
- Questions: [Ask a question](https://github.com/picselliahq/picsellia-cv-pipelines/issues/new)

**Email Support**
- Technical support: support@picsellia.com
- General inquiries: hello@picsellia.com

**Official Resources**
- [Picsellia CV Engine Docs](https://picselliahq.github.io/picsellia-cv-engine)
- [Picsellia Platform Docs](https://picsellia.com/docs)
- [Picsellia Website](https://picsellia.com)

---

## 📊 Repository Statistics

- **Total Pipelines**: 17 production-ready
- **Documentation Pages**: 20+ (guides + pipeline READMEs)
- **Lines of Documentation**: 12,000+
- **Code Examples**: 100+
- **Configuration Templates**: 50+

---

## 🗺️ Documentation Versions

- **Current Version**: 1.0.0 (2026-01-08)
- **Last Updated**: January 8, 2026
- **Maintained By**: Picsellia Team

---

## 💡 Tips for Success

1. **Start simple** - Test with default parameters first
2. **Read pipeline READMEs** - They contain crucial details
3. **Use local testing** - Validate before deploying
4. **Check examples** - Copy-paste and modify
5. **Ask for help** - Use GitHub Issues or email support

---

**Built with ❤️ by the Picsellia Team**

[Website](https://picsellia.com) | [Documentation](https://picsellia.com/docs) | [Blog](https://picsellia.com/blog) | [GitHub](https://github.com/picselliahq)

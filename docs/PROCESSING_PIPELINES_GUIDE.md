# Processing Pipelines User Guide

Complete guide to using Picsellia CV processing pipelines with the `pxl-pipeline` CLI.

---

## Table of Contents

- [Overview](#overview)
- [Pre-Annotation Pipelines](#pre-annotation-pipelines)
- [Dataset Processing Pipelines](#dataset-processing-pipelines)
- [Common Workflows](#common-workflows)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

Processing pipelines transform datasets without training models. They fall into two categories:

**1. Pre-Annotation (`PRE_ANNOTATION`)**
- Generate annotations automatically using foundation models
- Bootstrap dataset labeling
- Reduce manual annotation time by 70-90%

**2. Dataset Processing (`DATASET_VERSION_CREATION`)**
- Transform images and annotations
- Augment data
- Create derived dataset versions

---

## Pre-Annotation Pipelines

### SAM-3 Bounding Box Pre-annotation

**Purpose:** Zero-shot object detection using natural language prompts.

**Use Cases:**
- Multi-class object detection
- Bootstrapping new datasets
- Quality control verification

#### Quick Start

**1. Create run configuration:**

```toml
[job]
type = "PRE_ANNOTATION"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
text_prompt = "car,person,bicycle"
threshold = 0.35
mask_threshold = 0.5
iou_threshold = 0.5
containment_threshold = 0.8
deduplication_strategy = "keep_smaller"
min_area = 100.0

[input.dataset_version]
id = "your-dataset-id"

[input.model_version]
id = "sam3-model-id"
```

**2. Test locally:**

```bash
cd pipelines/SAM3_Bbox
pxl-pipeline test SAM3_Bbox --run-config-file run_config.toml
```

**3. Review results in Picsellia UI**

**4. Deploy for production:**

```bash
pxl-pipeline deploy SAM3_Bbox --organization my-org --bump patch
```

For detailed parameter documentation, see the [SAM3_Bbox README](../pipelines/SAM3_Bbox/README.md).

---

### SAM-3 Polygon Pre-annotation

**Purpose:** Zero-shot segmentation with precise polygon masks.

**Use Cases:**
- Instance segmentation datasets
- Medical imaging segmentation
- Precise object boundaries

See the [SAM3_polygons README](../pipelines/SAM3_polygons/README.md) for full documentation.

---

### Other Pre-Annotation Pipelines

**Grounding DINO** - Text-guided zero-shot detection
**YOLOv8 Pre-annotation** - Fast detection with COCO pretrained weights

See respective pipeline README files for detailed usage.

---

## Dataset Processing Pipelines

### Dataset Tiler

**Purpose:** Split large images into smaller tiles with overlap.

#### Quick Start

```bash
cd pipelines/dataset_tiler
pxl-pipeline test dataset_tiler --run-config-file run_config.toml
```

See [Main README](../README.md#dataset-processing-pipelines) for full documentation.

---

## Common Workflows

See [Main README - Common Workflows](../README.md#batch-operations) for complete workflow examples.

---

## Configuration Reference

See [Main README - Pipeline Configuration](../README.md#pipeline-configuration) for configuration format.

---

## Troubleshooting

See [Main README - Troubleshooting](../README.md#troubleshooting) for common issues and solutions.

---

**For complete documentation, see:**
- [Main README](../README.md)
- [Training Pipelines Guide](TRAINING_PIPELINES_GUIDE.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- Pipeline-specific README files in `pipelines/` directories

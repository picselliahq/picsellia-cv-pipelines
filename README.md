# Picsellia CV Pipelines

**The official public repository for computer vision pipelines powered by [Picsellia CV Engine](https://picselliahq.github.io/picsellia-cv-engine).**

This repository centralizes 17 production-ready pipelines for processing datasets and training computer vision models, all managed through the `pxl-pipeline` CLI and fully integrated with the Picsellia platform.

---

## 🚀 Quick Start

### Installation

```bash
pip install picsellia-cv-engine picsellia-pipelines-cli
```

### Authentication

```bash
export PXL_API_TOKEN="your-token"
pxl-pipeline login --organization my-org --env PROD --token $PXL_API_TOKEN
```

### Test a Pipeline

```bash
cd pipelines/SAM3_Bbox
pxl-pipeline test SAM3_Bbox --run-config-file run_config.toml
```

### Deploy to Picsellia

```bash
pxl-pipeline deploy SAM3_Bbox --organization my-org --bump patch
```

---

## 📦 What's Inside

### 🎯 Pre-Annotation Pipelines
Automatically generate annotations using foundation models:
- **SAM-3 Bbox/Polygons** - Zero-shot segmentation with natural language prompts
- **Grounding DINO** - Text-guided object detection
- **YOLOv8 Pre-annotation** - Fast detection with pretrained weights

### 🚂 Training Pipelines
Fine-tune models on custom datasets:
- **YOLOv8** - Real-time object detection (3 variants)
- **RT-DETR** - Transformer-based detection
- **YOLOv7 Segmentation** - Instance segmentation
- **SAM-2, CLIP, DINOv2** - Foundation models
- **Paddle OCR** - Text recognition

### 🔧 Dataset Processing Pipelines
Transform and prepare datasets:
- **Dataset Tiler** - Split large images with overlap handling
- **Albumentations** - Image augmentation
- **Bounding Box Cropper** - Extract regions
- **Diversified Extractor** - Sample representative data

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[Processing Pipelines Guide](docs/PROCESSING_PIPELINES_GUIDE.md)** | Pre-annotation and dataset processing |
| **[Training Pipelines Guide](docs/TRAINING_PIPELINES_GUIDE.md)** | Training custom models |
| **[Developer Guide](docs/DEVELOPER_GUIDE.md)** | Creating custom pipelines |
| **[Documentation Index](docs/README.md)** | Full documentation navigation |

### Pipeline-Specific Documentation

Each pipeline has its own detailed README:
- [`pipelines/SAM3_Bbox/README.md`](pipelines/SAM3_Bbox/README.md) - Comprehensive parameter guide
- [`pipelines/SAM3_polygons/README.md`](pipelines/SAM3_polygons/README.md) - Polygon segmentation
- [`pipelines/rt_detr/README.md`](pipelines/rt_detr/README.md) - RT-DETR training
- And more in each pipeline directory

---

## 💡 Usage Examples

### Pre-annotate a Dataset

```bash
# 1. Create configuration
cat > run_config.toml << EOF
[job]
type = "PRE_ANNOTATION"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
text_prompt = "car,person,bicycle"
threshold = 0.35

[input.dataset_version]
id = "your-dataset-id"
EOF

# 2. Test locally
pxl-pipeline test SAM3_Bbox --run-config-file run_config.toml

# 3. Deploy
pxl-pipeline deploy SAM3_Bbox --organization my-org --bump patch
```

### Train a Custom Detector

```bash
# 1. Create training config
cat > train_config.toml << EOF
[job]
type = "TRAINING"

[auth]
organization_name = "my-org"
env = "PROD"

[parameters]
epochs = 100
batch_size = 16
learning_rate = 0.001

[input.dataset_collection.train]
id = "train-id"

[input.dataset_collection.val]
id = "val-id"

[input.model_version]
id = "pretrained-id"
EOF

# 2. Test
cd pipelines/yolov8/training
pxl-pipeline test training --run-config-file train_config.toml

# 3. Deploy for full training
pxl-pipeline deploy training --organization my-org --bump minor
```

### Process Large Images

```bash
# Tile satellite/medical imagery
cd pipelines/dataset_tiler
pxl-pipeline test dataset_tiler --run-config-file tile_config.toml
```

---

## 🔧 `pxl-pipeline` CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `login` | Authenticate with Picsellia | `pxl-pipeline login --organization my-org --token $TOKEN` |
| `init` | Create new pipeline from template | `pxl-pipeline init my_pipeline --type processing --template pre_annotation` |
| `test` | Test pipeline locally | `pxl-pipeline test <pipeline> --run-config-file config.toml` |
| `deploy` | Deploy to Picsellia cloud | `pxl-pipeline deploy <pipeline> --organization my-org --bump patch` |

**Learn more:** [Picsellia CV Engine - CLI Usage](https://picselliahq.github.io/picsellia-cv-engine/usage/)

---

## 🏗️ Repository Structure

```
picsellia-cv-pipelines/
├── pipelines/              # 17 production-ready pipelines
│   ├── SAM3_Bbox/
│   ├── yolov8/
│   ├── rt_detr/
│   └── ...
├── docs/                   # Comprehensive guides
│   ├── PROCESSING_PIPELINES_GUIDE.md
│   ├── TRAINING_PIPELINES_GUIDE.md
│   └── DEVELOPER_GUIDE.md
├── tests/                  # Test configurations
├── scripts/                # Automation scripts
└── .github/workflows/      # CI/CD
```

**Each pipeline includes:**
- `pipeline.py` - Main entry point
- `steps.py` - Step implementations
- `utils/parameters.py` - Parameter definitions
- `config.toml` - Metadata
- `Dockerfile` - Container definition
- `README.md` - Detailed documentation

---

## 🎓 Getting Started Path

**For Users:**
1. Install CLI tools → See [Quick Start](#quick-start)
2. Choose your use case:
   - **Processing data?** → [Processing Guide](docs/PROCESSING_PIPELINES_GUIDE.md)
   - **Training models?** → [Training Guide](docs/TRAINING_PIPELINES_GUIDE.md)
3. Check pipeline-specific README for detailed parameters

**For Developers:**
1. Clone repository: `git clone https://github.com/picselliahq/picsellia-cv-pipelines.git`
2. Setup: `uv sync && pxl-pipeline login`
3. Read: [Developer Guide](docs/DEVELOPER_GUIDE.md)
4. Create pipeline: `pxl-pipeline init my_pipeline --type processing`

---

## 🛠️ Development

### Quick Development Setup

```bash
# Clone and setup
git clone https://github.com/picselliahq/picsellia-cv-pipelines.git
cd picsellia-cv-pipelines
uv sync
pre-commit install

# Authenticate
pxl-pipeline login --organization test-account --env STAGING --token $TOKEN
```

### Create a Custom Pipeline

```bash
# Initialize from template
pxl-pipeline init my_pipeline --type processing --template pre_annotation

# Customize
cd pipelines/my_pipeline
# Edit utils/parameters.py, steps.py, pipeline.py

# Test
pxl-pipeline test my_pipeline --run-config-file test_config.toml

# Deploy
pxl-pipeline deploy my_pipeline --organization my-org --bump patch
```

**Full guide:** [Developer Guide](docs/DEVELOPER_GUIDE.md)

---

## 🔬 Testing & CI/CD

### Run Tests

```bash
# Test specific pipeline
./scripts/test_pipelines.sh --pipeline SAM3_Bbox

# Test all pipelines
./scripts/test_pipelines.sh --pipeline all
```

### CI/CD

- **Automated testing** on every PR
- **GPU tests** on self-hosted runners
- **Code quality checks** with Ruff

---

## 💻 Hardware Requirements

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| **Processing (CPU)** | 4 cores, 8GB RAM | 8 cores, 16GB RAM |
| **Processing (GPU)** | 8GB VRAM | 16GB VRAM (10-20x faster) |
| **Training** | 8GB VRAM, 16GB RAM | 24GB VRAM, 32GB RAM, NVMe SSD |

---

## 🆘 Troubleshooting

### Common Issues

**Authentication errors?**
```bash
pxl-pipeline login --organization my-org --token $NEW_TOKEN
```

**Out of memory?**
```toml
# Reduce batch size in config
[parameters]
batch_size = 4
```

**Slow processing?**
```bash
# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
```

**More help:** [Troubleshooting sections in guides](docs/)

---

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes (follow existing patterns)
4. Add tests and documentation
5. Submit a pull request

**Contribution types:**
- New pipelines
- Bug fixes
- Documentation improvements
- Performance optimizations

**See:** [Contributing section in docs](docs/DEVELOPER_GUIDE.md#contributing)

---

## 📝 License

Apache License 2.0 - See [LICENSE](LICENSE)

Individual pipelines may use models with different licenses. Check model cards on Hugging Face and framework licenses.

---

## 🙏 Acknowledgments

Built with:
- **[Picsellia CV Engine](https://picselliahq.github.io/picsellia-cv-engine)** - Core framework
- **Meta AI** - SAM-3, SAM-2, DINOv2
- **Hugging Face** - Model hosting and transformers
- **Ultralytics** - YOLOv8, YOLOv7
- **OpenAI** - CLIP
- **PaddlePaddle** - Paddle OCR

---

## 🔗 Links

- **Documentation**: [Full Documentation Index](docs/README.md)
- **CV Engine Docs**: [picselliahq.github.io/picsellia-cv-engine](https://picselliahq.github.io/picsellia-cv-engine)
- **Picsellia Platform**: [picsellia.com](https://picsellia.com)
- **GitHub Issues**: [Report bugs or request features](https://github.com/picselliahq/picsellia-cv-pipelines/issues)
- **Support**: support@picsellia.com

---

**Built by Picsellia** | [Website](https://picsellia.com) | [Documentation](https://picsellia.com/docs) | [Blog](https://picsellia.com/blog)

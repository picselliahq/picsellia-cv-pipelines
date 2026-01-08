# Developer Guide

Complete guide for developing custom pipelines using the Picsellia CV Engine and `pxl-pipeline` CLI.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Creating a New Pipeline](#creating-a-new-pipeline)
- [Pipeline Architecture](#pipeline-architecture)
- [Parameter System](#parameter-system)
- [Step Development](#step-development)
- [Testing Locally](#testing-locally)
- [Docker Containerization](#docker-containerization)
- [Deployment](#deployment)
- [Best Practices](#best-practices)
- [Advanced Topics](#advanced-topics)

---

## Getting Started

### Prerequisites

- Python 3.10-3.13
- pip or uv package manager
- Docker (for deployment)
- Picsellia account with API token
- Basic understanding of computer vision and Python

### Development Environment Setup

```bash
# Clone repository
git clone https://github.com/picselliahq/picsellia-cv-pipelines.git
cd picsellia-cv-pipelines

# Install dependencies
uv sync

# Install CLI tools
uv pip install picsellia-pipelines-cli picsellia-cv-engine

# Install pre-commit hooks
pre-commit install

# Authenticate
export PXL_API_TOKEN="your-token"
pxl-pipeline login --organization test-account --env STAGING --token $PXL_API_TOKEN
```

### Repository Structure

```
picsellia-cv-pipelines/
├── pipelines/           # All pipeline implementations
├── tests/              # Test configurations
├── scripts/            # Automation scripts
├── docs/               # Documentation
└── .github/workflows/  # CI/CD
```

---

## Creating a New Pipeline

### Initialize from Template

The `pxl-pipeline init` command creates a new pipeline from built-in templates.

**Processing Pipeline:**

```bash
pxl-pipeline init my_processing_pipeline \
    --type processing \
    --template pre_annotation \
    --output-dir pipelines/
```

**Training Pipeline:**

```bash
pxl-pipeline init my_training_pipeline \
    --type training \
    --template yolov8 \
    --output-dir pipelines/
```

### Generated Structure

```
pipelines/my_pipeline/
├── config.toml           # Pipeline metadata
├── pipeline.py           # Entry point
├── steps.py              # Step implementations
├── utils/
│   ├── parameters.py     # Parameter classes
│   └── processing.py     # Business logic
├── pyproject.toml        # Dependencies
├── Dockerfile            # Container definition
├── .dockerignore         # Docker ignore rules
└── .venv/                # Virtual environment (created by uv)
```

### Manual Creation

If you prefer not to use templates:

**1. Create directory:**

```bash
mkdir -p pipelines/my_pipeline/utils
cd pipelines/my_pipeline
```

**2. Create `config.toml`:**

```toml
[metadata]
name = "my_pipeline"
version = "0.1.0"
description = "My custom pipeline"
type = "PRE_ANNOTATION"  # or TRAINING, DATASET_VERSION_CREATION

[execution]
pipeline_script = "pipeline.py"
requirements_file = "pyproject.toml"
parameters_class = "utils/parameters.py:ProcessingParameters"

[docker]
image_name = "picsellia/my-pipeline"
image_tag = "latest"
cpu = 4
gpu = 1
```

**3. Create `pyproject.toml`:**

```toml
[project]
name = "my-pipeline"
version = "0.1.0"
requires-python = ">=3.10,<3.14"
dependencies = [
    "picsellia-cv-engine>=0.4.1",
    "picsellia-pipelines-cli",
    # Add your dependencies
]
```

**4. Initialize environment:**

```bash
uv sync
```

---

## Pipeline Architecture

### Core Components

**1. Context**
- Manages execution mode (local vs Picsellia)
- Provides access to parameters, datasets, models
- Handles authentication and API communication

**2. Decorators**
- `@pipeline`: Marks main pipeline function
- `@step`: Marks reusable step functions

**3. Parameters**
- Type-safe configuration
- Validation and defaults
- `Parameters` (processing) or `HyperParameters` (training)

**4. Steps**
- Composable, reusable functions
- Input/output typed
- Can be framework-specific or generic

### Pipeline Flow

```
Context Creation
    ↓
@pipeline function
    ↓
@step functions (orchestrated)
    ↓
Results (uploaded to Picsellia)
```

---

## Parameter System

### Processing Parameters

Create `utils/parameters.py`:

```python
from picsellia_cv_engine.core.parameters import Parameters
from typing import Union

class ProcessingParameters(Parameters):
    """Custom processing parameters"""
    
    def __init__(self, log_data):
        super().__init__(log_data=log_data)
        
        # Required parameter
        self.text_prompt = self.extract_parameter(
            keys=["text_prompt"],
            expected_type=str,
            required=True,
            description="Comma-separated class names"
        )
        
        # Float with range
        self.threshold = self.extract_parameter(
            keys=["threshold", "conf"],
            expected_type=float,
            default=0.5,
            range_value=(0.0, 1.0),
            description="Detection confidence threshold"
        )
        
        # Integer parameter
        self.batch_size = self.extract_parameter(
            keys=["batch_size"],
            expected_type=int,
            default=8,
            range_value=(1, 128),
            description="Processing batch size"
        )
        
        # String with choices
        self.strategy = self.extract_parameter(
            keys=["strategy"],
            expected_type=str,
            default="keep_smaller",
            description="Deduplication strategy"
        )
        
        # Boolean parameter
        self.use_gpu = self.extract_parameter(
            keys=["use_gpu"],
            expected_type=bool,
            default=True,
            description="Use GPU if available"
        )
        
        # List parameter
        self.class_names = self.extract_parameter(
            keys=["class_names"],
            expected_type=list,
            default=[],
            description="List of class names"
        )
        
    def validate(self):
        """Custom validation logic"""
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
```

### Training Parameters

```python
from picsellia_cv_engine.core.parameters import HyperParameters

class TrainingHyperParameters(HyperParameters):
    """Custom training hyperparameters"""
    
    def __init__(self, log_data):
        super().__init__(log_data=log_data)
        
        self.epochs = self.extract_parameter(
            keys=["epochs"],
            expected_type=int,
            default=100,
            description="Number of training epochs"
        )
        
        self.learning_rate = self.extract_parameter(
            keys=["learning_rate", "lr"],
            expected_type=float,
            default=0.001,
            range_value=(1e-6, 1.0),
            description="Learning rate"
        )
        
        self.batch_size = self.extract_parameter(
            keys=["batch_size"],
            expected_type=int,
            default=16,
            description="Batch size"
        )
        
        self.patience = self.extract_parameter(
            keys=["patience"],
            expected_type=int,
            default=20,
            description="Early stopping patience"
        )
```

### Parameter Best Practices

1. **Use descriptive names**: `detection_threshold` not `thresh`
2. **Provide defaults**: Always have sensible defaults
3. **Add descriptions**: Help users understand parameters
4. **Validate ranges**: Use `range_value` for numeric parameters
5. **Support aliases**: Use `keys=["learning_rate", "lr"]` for common variations
6. **Document in README**: Explain each parameter's effect

---

## Step Development

### Step Anatomy

A step is a function decorated with `@step` that performs a specific operation.

```python
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.core.types import CocoDataset
from typing import Dict, Any

@step
def my_processing_step(
    dataset: CocoDataset,
    model: Any = None
) -> CocoDataset:
    """
    Process dataset using model.
    
    Args:
        dataset: Input COCO dataset
        model: Optional model for processing
        
    Returns:
        Processed COCO dataset
    """
    # Access context and parameters
    from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
    context = Pipeline.get_active_context()
    params = context.processing_parameters
    
    # Your processing logic
    for image in dataset.images:
        # Process each image
        pass
    
    return dataset
```

### Generic Steps

Use built-in steps from the CV Engine:

```python
from picsellia_cv_engine.steps import (
    load_coco_datasets,
    upload_dataset_annotations,
    upload_full_dataset,
    validate_dataset,
)
```

**Common Generic Steps:**

| Step | Purpose | Returns |
|------|---------|---------|
| `load_coco_datasets()` | Load datasets from Picsellia | `CocoDataset` or `DatasetCollection` |
| `upload_dataset_annotations()` | Upload annotations to Picsellia | None |
| `upload_full_dataset()` | Upload full dataset (images + annotations) | None |
| `validate_dataset()` | Validate dataset format | None |

### Framework-Specific Steps

For frameworks like Ultralytics, use framework-specific steps:

```python
from picsellia_cv_engine.frameworks.ultralytics.steps import (
    load_ultralytics_model,
    train_ultralytics_model,
    evaluate_ultralytics_model,
    export_ultralytics_model,
)
```

### Custom Step Example

Create `steps.py`:

```python
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.core.types import CocoDataset
import torch
from transformers import AutoModel, AutoProcessor

@step
def load_my_model():
    """Load custom model from HuggingFace."""
    context = Pipeline.get_active_context()
    params = context.processing_parameters
    
    # Load model
    model = AutoModel.from_pretrained(params.model_name)
    processor = AutoProcessor.from_pretrained(params.model_name)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() and params.use_gpu else "cpu"
    model = model.to(device)
    model.eval()
    
    return {"model": model, "processor": processor, "device": device}

@step
def process_with_model(
    dataset: CocoDataset,
    model_dict: dict
) -> CocoDataset:
    """Run inference on dataset."""
    context = Pipeline.get_active_context()
    params = context.processing_parameters
    
    model = model_dict["model"]
    processor = model_dict["processor"]
    device = model_dict["device"]
    
    # Process each image
    for image_info in dataset.images:
        # Load image
        image_path = image_info["file_path"]
        image = load_image(image_path)
        
        # Preprocess
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process
        predictions = postprocess_outputs(outputs)
        
        # Add to dataset
        for pred in predictions:
            dataset.add_annotation({
                "image_id": image_info["id"],
                "category_id": pred["category_id"],
                "bbox": pred["bbox"],
                "score": pred["score"],
            })
    
    return dataset
```

---

## Pipeline Implementation

### Processing Pipeline Example

Create `pipeline.py`:

```python
import argparse
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.core.enums import ProcessingType
from picsellia_cv_engine.steps import (
    load_coco_datasets,
    upload_dataset_annotations,
)
from steps import load_my_model, process_with_model
from utils.parameters import ProcessingParameters

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

# Create context
context = create_processing_context_from_config(
    processing_type=ProcessingType.PRE_ANNOTATION,
    processing_parameters_cls=ProcessingParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)

# Define pipeline
@pipeline(context=context, log_folder_path="logs/", remove_logs_on_completion=False)
def my_pipeline():
    """My custom pre-annotation pipeline."""
    # Load data
    dataset = load_coco_datasets()
    
    # Load model
    model_dict = load_my_model()
    
    # Process
    processed_dataset = process_with_model(
        dataset=dataset,
        model_dict=model_dict
    )
    
    # Upload results
    upload_dataset_annotations(
        dataset=processed_dataset,
        use_id=True
    )

if __name__ == "__main__":
    my_pipeline()
```

### Training Pipeline Example

```python
import argparse
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.core.services.context.unified_context import (
    create_training_context_from_config,
)
from picsellia_cv_engine.steps import load_coco_datasets
from picsellia_cv_engine.frameworks.ultralytics.steps import (
    prepare_ultralytics_dataset,
    load_ultralytics_model,
    train_ultralytics_model,
    evaluate_ultralytics_model,
    export_ultralytics_model,
)
from utils.parameters import TrainingHyperParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_training_context_from_config(
    training_parameters_cls=TrainingHyperParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)

@pipeline(context=context, log_folder_path="logs/")
def training_pipeline():
    """My custom training pipeline."""
    # Load datasets
    dataset_collection = load_coco_datasets()
    
    # Prepare for Ultralytics
    dataset_collection = prepare_ultralytics_dataset(
        dataset_collection=dataset_collection
    )
    
    # Load pretrained model
    model = load_ultralytics_model(
        pretrained_weights_name="pretrained-weights"
    )
    
    # Train
    model = train_ultralytics_model(
        model=model,
        dataset_collection=dataset_collection
    )
    
    # Export
    export_ultralytics_model(model=model)
    
    # Evaluate
    evaluate_ultralytics_model(
        model=model,
        dataset=dataset_collection["test"]
    )

if __name__ == "__main__":
    training_pipeline()
```

---

## Testing Locally

### Create Test Configuration

Create `test_config.toml`:

```toml
[job]
type = "PRE_ANNOTATION"

[auth]
organization_name = "test-account"
env = "STAGING"

[parameters]
text_prompt = "car,person"
threshold = 0.5
batch_size = 8

[input.dataset_version]
id = "test-dataset-id"

[input.model_version]
id = "test-model-id"
```

### Run Local Test

```bash
cd pipelines/my_pipeline
pxl-pipeline test my_pipeline --run-config-file test_config.toml
```

### Test with Python Directly

```bash
python pipeline.py --mode local --config-file test_config.toml
```

### Debug Mode

Add debugging to your pipeline:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

@pipeline(context=context, log_folder_path="logs/")
def my_pipeline():
    logging.debug("Starting pipeline")
    dataset = load_coco_datasets()
    logging.debug(f"Loaded {len(dataset.images)} images")
    # ...
```

### Reuse Run Directory

Skip re-downloading data:

```bash
pxl-pipeline test my_pipeline \
    --run-config-file test_config.toml \
    --reuse-dir
```

---

## Docker Containerization

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM picsellia/cuda:11.8.0-cudnn8-ubuntu22.04-python3.10

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /experiment

# Clone base Docker utilities
RUN git clone --depth 1 https://github.com/picselliahq/picsellia-cv-base-docker.git /tmp/base-docker && \
    cp -r /tmp/base-docker/base/. /experiment && \
    rm -rf /tmp/base-docker

# Activate venv in run.sh
RUN sed -i '1 a source /experiment/my_pipeline/.venv/bin/activate' /experiment/run.sh

# Copy and sync dependencies
COPY ./uv.lock my_pipeline/uv.lock
COPY ./pyproject.toml my_pipeline/pyproject.toml
RUN uv sync --python=$(which python3.10) --project my_pipeline --prerelease allow

# Copy source code
COPY ./ my_pipeline/

# Set environment variables
ENV PYTHONPATH="/experiment"
ENV HF_HOME="/experiment/.cache/huggingface"
ENV TRANSFORMERS_CACHE="${HF_HOME}/transformers"

ENTRYPOINT ["run", "python3.10", "my_pipeline/pipeline.py"]
```

### .dockerignore

Create `.dockerignore`:

```
.venv/
venv/
__pycache__/
*.pyc
*.pyo
.DS_Store
logs/
runs/
*.log
.git/
.github/
tests/
```

### Build and Test

```bash
cd pipelines/my_pipeline

# Build image
docker build -t picsellia/my-pipeline:latest .

# Test locally
docker run -v $(pwd)/runs:/experiment/my_pipeline/runs \
    picsellia/my-pipeline:latest \
    --mode local --config-file runs/test_config.toml
```

---

## Deployment

### Configure Docker Settings

Update `config.toml`:

```toml
[docker]
image_name = "picsellia/my-pipeline"
image_tag = "latest"
cpu = 4
gpu = 1  # 0 for CPU-only pipelines
```

### Deploy Pipeline

```bash
cd pipelines/my_pipeline

pxl-pipeline deploy my_pipeline \
    --organization my-org \
    --env PROD \
    --bump patch
```

**Version bumps:**
- `patch`: Bug fixes (0.0.X)
- `minor`: New features (0.X.0)
- `major`: Breaking changes (X.0.0)
- `rc`: Release candidate
- `final`: Production release

### Deployment Process

The `pxl-pipeline deploy` command:

1. Reads `config.toml`
2. Builds Docker image
3. Tags with version and environment
4. Pushes to registry
5. Registers pipeline in Picsellia
6. Sets resource requirements

### Batch Deployment

Use the provided script:

```bash
./scripts/deploy_pipelines.sh --pipeline my_pipeline \
    --organization my-org \
    --env STAGING \
    --bump minor \
    --token $PXL_API_TOKEN
```

---

## Best Practices

### Code Organization

1. **Separate concerns**: Parameters, steps, business logic
2. **Modular steps**: Small, testable, reusable
3. **Type hints**: Use typing for clarity
4. **Documentation**: Docstrings for all functions
5. **Error handling**: Graceful failures with clear messages

### Performance

1. **Batch processing**: Process multiple images at once
2. **GPU utilization**: Use CUDA when available
3. **Memory management**: Clear caches, delete unused objects
4. **Parallel loading**: Use workers for data loading
5. **Cache models**: Download once, reuse

### Testing

1. **Unit tests**: Test individual steps
2. **Integration tests**: Test full pipeline
3. **Edge cases**: Test empty data, invalid inputs
4. **Performance tests**: Measure speed and memory
5. **CI/CD**: Automate testing

### Documentation

1. **README**: Pipeline overview, parameters, examples
2. **Inline comments**: Explain complex logic
3. **Docstrings**: All functions and classes
4. **Examples**: Show common use cases
5. **Troubleshooting**: Common issues and solutions

### Version Control

1. **Semantic versioning**: Major.minor.patch
2. **Changelog**: Document changes
3. **Git tags**: Tag releases
4. **Branching**: Feature branches for development
5. **Pull requests**: Review before merging

---

## Advanced Topics

### Custom Context

For advanced use cases, create custom contexts:

```python
from picsellia_cv_engine.core.services.context.processing_context import ProcessingContext

class MyCustomContext(ProcessingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization
        self.custom_attribute = "value"
    
    def custom_method(self):
        # Custom logic
        pass
```

### Callbacks and Hooks

Add callbacks for training:

```python
from ultralytics.utils.callbacks import Callbacks

def on_train_epoch_end(trainer):
    # Custom logic after each epoch
    print(f"Epoch {trainer.epoch} completed")

callbacks = Callbacks()
callbacks.register_action("on_train_epoch_end", on_train_epoch_end)
```

### Multi-GPU Training

For multiple GPUs:

```python
import torch.distributed as dist

# In your training step
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Custom Data Loaders

Implement custom data loading:

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, coco_dataset):
        self.coco_dataset = coco_dataset
    
    def __len__(self):
        return len(self.coco_dataset.images)
    
    def __getitem__(self, idx):
        # Custom loading logic
        pass

# In your step
dataloader = DataLoader(
    CustomDataset(dataset),
    batch_size=params.batch_size,
    num_workers=params.workers
)
```

### Model Ensembling

Combine multiple models:

```python
@step
def ensemble_predictions(
    dataset: CocoDataset,
    models: list
) -> CocoDataset:
    """Ensemble predictions from multiple models."""
    all_predictions = []
    
    for model in models:
        predictions = run_inference(model, dataset)
        all_predictions.append(predictions)
    
    # Combine predictions (e.g., voting, averaging)
    final_predictions = combine_predictions(all_predictions)
    
    return create_dataset_from_predictions(dataset, final_predictions)
```

### Distributed Processing

For large-scale processing:

```python
from multiprocessing import Pool

@step
def process_in_parallel(dataset: CocoDataset) -> CocoDataset:
    """Process images in parallel."""
    with Pool(processes=8) as pool:
        results = pool.map(process_single_image, dataset.images)
    
    # Combine results
    return create_dataset_from_results(results)
```

---

## Troubleshooting Development

### Issue: Import Errors

```bash
# Ensure packages are installed
cd pipelines/my_pipeline
uv sync

# Check Python path
python -c "import sys; print(sys.path)"

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Context Not Found

```python
# Ensure context is created before @pipeline decorator
context = create_processing_context_from_config(...)

@pipeline(context=context, ...)
def my_pipeline():
    pass
```

### Issue: Parameters Not Loading

```python
# Debug parameter extraction
def __init__(self, log_data):
    super().__init__(log_data=log_data)
    print(f"Available parameters: {log_data}")
    self.my_param = self.extract_parameter(...)
```

### Issue: Docker Build Failures

```bash
# Check Dockerfile syntax
docker build --no-cache -t test .

# Verify COPY paths
# Ensure files exist before COPY

# Check base image
docker pull picsellia/cuda:11.8.0-cudnn8-ubuntu22.04-python3.10
```

---

## Contributing

### Contribution Workflow

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Run linting: `ruff check . --fix`
6. Commit with clear message
7. Push and create PR

### Code Style

Use Ruff for consistent styling:

```bash
# Check code
ruff check .

# Fix issues
ruff check --fix .

# Format
ruff format .
```

### Pull Request Checklist

- [ ] Code passes linting
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Tested locally
- [ ] CI/CD passes

---

## Resources

- **[Picsellia CV Engine Docs](https://picselliahq.github.io/picsellia-cv-engine)** - Official documentation
- **[Processing Guide](PROCESSING_PIPELINES_GUIDE.md)** - Using processing pipelines
- **[Training Guide](TRAINING_PIPELINES_GUIDE.md)** - Training models
- **[GitHub Issues](https://github.com/picselliahq/picsellia-cv-pipelines/issues)** - Report bugs, request features

---

**Happy Developing! 🚀**

For questions or support: support@picsellia.com

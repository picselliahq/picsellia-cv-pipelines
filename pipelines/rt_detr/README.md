# RT-DETR Training Pipeline

## Overview

This pipeline enables fine-tuning of **RT-DETR** (Real-Time Detection Transformer) models for object detection tasks on your custom datasets using the Picsellia platform. RT-DETR is a state-of-the-art real-time object detector that combines the accuracy of transformer-based models with efficient inference speed.

## Model Card

### Supported Models

The pipeline supports RT-DETR models from the Hugging Face Hub. The default model is:

- **PekingU/rtdetr_v2_r50vd** (ResNet-50 backbone)

Other compatible RT-DETR variants include:
- `PekingU/rtdetr_v2_r18vd` (lighter, faster)
- `PekingU/rtdetr_v2_r34vd` (medium-weight)
- `PekingU/rtdetr_v2_r101vd` (heavier, more accurate)

### Model Architecture

RT-DETR uses a transformer-based architecture optimized for real-time inference:
- **Backbone**: ResNet variants (R18, R34, R50, R101) with deformable convolutions
- **Encoder-Decoder**: Efficient transformer with hybrid encoder and decoder
- **Detection Head**: End-to-end detection without NMS post-processing

### Use Cases

- Real-time object detection applications
- Industrial quality control
- Surveillance and security systems
- Autonomous vehicles
- Retail analytics
- Custom object detection with training on your labeled datasets

## Pipeline Steps

The training pipeline consists of two main steps:

1. **Training**: Fine-tunes the RT-DETR model on your COCO-formatted datasets
2. **Evaluation**: Runs inference on the test split and evaluates performance in Picsellia

## Parameters

### Training Hyperparameters

Configure these parameters in your Picsellia experiment:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | `3` | Number of training epochs |
| `batch_size` | int | `8` | Batch size for training (per device) |
| `image_size` | int | `640` | Input image size for training |
| `model_name` or `repo_id` | str | `"PekingU/rtdetr_v2_r50vd"` | Hugging Face model identifier |
| `learning_rate` | float | `5e-5` | Learning rate for optimizer |
| `weight_decay` | float | `0.05` | Weight decay for regularization |
| `warmup_ratio` | float | `0.05` | Ratio of training steps for learning rate warmup |

### Advanced Training Arguments

Additional parameters controlled via environment variables:

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `WORKERS` | int | `4` | Number of data loading workers |

### Evaluation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conf_thresh` | float | `0.25` | Confidence threshold for inference |

## Dataset Requirements

### Format
The pipeline expects datasets in **COCO format** with three splits:
- `train`: Training dataset
- `val`: Validation dataset
- `test`: Test dataset for evaluation

### Structure
Each dataset should contain:
- Images directory with all image files
- COCO annotation JSON file with:
  - Image metadata
  - Bounding box annotations (x, y, width, height)
  - Category labels
  - Optional: `iscrowd` flags (crowd annotations are excluded by default)

## Usage

### Running the Pipeline

#### On Picsellia Platform (Recommended)

1. Create an experiment in Picsellia
2. Attach your COCO-formatted datasets (train, val, test)
3. Attach a base model with pretrained weights (named `pretrained-weights`)
4. Configure hyperparameters in the experiment settings
5. Launch the training pipeline

#### Local Execution

```bash
python pipeline.py --mode local --config-file path/to/config.yaml
```

### Configuration Example

```yaml
hyperparameters:
  epochs: 10
  batch_size: 8
  image_size: 640
  model_name: "PekingU/rtdetr_v2_r50vd"
  learning_rate: 5e-5
  weight_decay: 0.05
  warmup_ratio: 0.05
```

## Outputs

### Artifacts

1. **Model Weights** (`model-latest`):
   - Zipped archive containing the final fine-tuned model
   - Includes both model weights and image processor configuration
   - Saved in Hugging Face format for easy deployment

2. **Training Logs**:
   - Loss curves
   - Learning rate schedule
   - Gradient norms
   - Validation loss
   - Evaluation runtime

### Evaluation Results

The evaluation step generates:
- Predictions on test set assets
- Performance metrics in Picsellia dashboard
- Visual evaluation results
- Comparison against ground truth annotations

## Hardware Requirements

### Recommended

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended for larger batches)
- **CPU**: Multi-core processor (4+ cores)
- **RAM**: 16GB+ system memory
- **Storage**: Sufficient space for datasets and model checkpoints

### Performance Optimization

- **Mixed Precision Training**: Automatically enabled when GPU is available (FP16)
- **Batch Size**: Adjust based on available VRAM
  - RTX 3090 (24GB): batch_size=16-32
  - RTX 3080 (10GB): batch_size=8-16
  - Tesla T4 (16GB): batch_size=8-16
- **Workers**: Increase for faster data loading (limited by CPU cores)

## Model Performance Tips

### For Better Accuracy
- Increase `epochs` (e.g., 50-100 for production models)
- Use larger backbone models (e.g., `rtdetr_v2_r101vd`)
- Increase `image_size` (e.g., 800-1024)
- Fine-tune `learning_rate` (try 1e-5 for transfer learning)

### For Faster Training
- Reduce `batch_size` if memory-constrained
- Use smaller backbone models (e.g., `rtdetr_v2_r18vd`)
- Reduce `image_size`
- Increase `workers` for data loading

### For Better Generalization
- Increase `weight_decay` (e.g., 0.1)
- Use data augmentation (configure in augmentation parameters)
- Ensure diverse training data
- Monitor validation loss to avoid overfitting

## Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors**
- Reduce `batch_size`
- Reduce `image_size`
- Use a smaller model variant
- Reduce `workers`

**Poor Performance**
- Increase `epochs`
- Verify dataset quality and annotations
- Adjust `learning_rate`
- Check class balance in your dataset

**Slow Training**
- Increase `batch_size` (if memory allows)
- Increase `workers`
- Verify GPU is being used (check logs)
- Use smaller `image_size` if acceptable

## Technical Details

### Training Process

1. **Dataset Loading**: COCO datasets are loaded and converted to Hugging Face format
2. **Model Initialization**: Pretrained RT-DETR model is loaded with custom label space
3. **Fine-tuning**: Model is trained using the Hugging Face Trainer
4. **Checkpointing**: Model is saved at each epoch
5. **Artifact Upload**: Final weights are zipped and uploaded to Picsellia

### Evaluation Process

1. **Model Loading**: Final trained weights are loaded
2. **Inference**: Model runs on test set images
3. **Post-processing**: Predictions are filtered by confidence threshold
4. **Validation**: Bounding boxes are clamped to image boundaries
5. **Metrics**: Results are evaluated and logged to Picsellia

## License & Attribution

This pipeline uses:
- **RT-DETR**: Models from PekingU/Baidu Research
- **Transformers**: Hugging Face library for model training
- **Picsellia CV Engine**: Core framework for pipeline orchestration

Refer to individual model cards on Hugging Face for specific model licenses.

## Support

For issues or questions:
- Check Picsellia documentation
- Contact Picsellia support
- Review logs in the `logs/` directory
- Verify dataset format and annotations

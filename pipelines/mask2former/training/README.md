# Mask2Former Training Pipeline (MASK mode)

**Train Mask2Former models on Picsellia `MASK` datasets (RLE-encoded masks) and register a `MASK` model version.**

Mask2Former (Hugging Face `transformers`, Swin-Tiny by default) is a query-based transformer: it predicts a fixed set of masks and classes, matched to the ground truth with a Hungarian assignment. No anchors, no NMS, no box head. It is the second `MASK` training pipeline, next to [`maskrcnn/training_mask`](../../maskrcnn/training_mask/README.md).

| | `maskrcnn/training_mask` | `mask2former/training` (this one) |
|---|---|---|
| Architecture | Mask R-CNN, ResNet-50 FPN (torchvision) | Mask2Former, Swin (Hugging Face) |
| Dataset version type | `MASK` | `MASK` |
| Model version `inference_type` | `MASK` | `MASK` |
| Optimizer | SGD | AdamW (backbone lr x0.1) |
| Export | TorchScript | PyTorch checkpoint only |
| Resizing | shorter side to `image_size` | stretched to `image_size` x `image_size` |

This pipeline only accepts `MASK` dataset versions, for the `train`, `val` and `test` splits alike: it stops at the start, with an explicit error, on any other type. For `SEGMENTATION` (polygon) datasets, convert them with the `polygons_to_RLE_mask` pipeline first.

## What You'll Get

- A Mask2Former model fine-tuned on your data, starting from a COCO instance-segmentation checkpoint with a class head re-created for your classes
- Experiment artifacts: `checkpoint-latest` (self-contained PyTorch checkpoint), `model-metadata`, `mask-metrics` and `mask-predictions` (JSON)
- The test-set predictions as Picsellia **evaluations** of the experiment (one per image, masks sent as RLE), with the metrics computed by the platform
- COCO metrics computed in the pipeline: mask and box mAP/mAR (`test/mAP50-95(M)`, ...) and a per-class table (`test/metrics`)
- A few test images with the predicted masks drawn on them (`test/prediction_NN`)
- Training curves (`train/*`, `val/*`), including the validation mask mAP per epoch

> **Note:** sending mask evaluations needs `picsellia>=6.37.0`. The pipeline needs network access to the Hugging Face Hub when it starts (`pretrained_model`).

## Dataset Configurations

Attach `MASK` dataset versions to the experiment with the aliases below. They are declared as dataset constraints in [`config.toml`](config.toml).

| Attached datasets | Aliases | How the splits are made |
|-------------------|---------|-------------------------|
| **1** | `train` | Random split into train / val / test: 80 % / 10 % / 10 % with the default `train_set_split_ratio` (0.8) |
| **2** | `train` and `test` | `train` is split into train / val (80 % / 20 %); `test` is used as it is |
| **3** | `train`, `val` and `test` | Each dataset is used as it is, nothing is split |

With two datasets, the second one must be called `test` (not `val`). The classes come from the `train` split.

## Quick Start

```toml
[hyperparameters]
epochs = 50
batch_size = 2
image_size = 512
pretrained_model = "facebook/mask2former-swin-tiny-coco-instance"
```

Mask2Former converges slowly: with few images or few epochs, the model may predict no instance at all (the pipeline then logs zero metrics and a warning instead of failing). For more accuracy, try a larger checkpoint (`facebook/mask2former-swin-small-coco-instance`, `...-swin-base-coco-instance`) and a larger `image_size`, at the price of more GPU memory.

## Parameters

The exhaustive list (defaults, meaning, outputs and logs) is the markdown description in [`config.toml`](config.toml), displayed on the model version in Picsellia. The main ones:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pretrained_model` | `facebook/mask2former-swin-tiny-coco-instance` | Hugging Face checkpoint to start from |
| `epochs` | `30` | Passes over the training split |
| `batch_size` | `2` | Images per iteration (lower it first on out-of-memory) |
| `image_size` | `512` | Square side the images are stretched to |
| `learning_rate` | `1e-4` | AdamW learning rate; the backbone uses `backbone_lr_multiplier` (0.1) times it |
| `clip_grad_norm` | `0.01` | Gradient clipping (official recipe) |
| `amp` | `true` | bfloat16 mixed precision on GPU; turn off if the loss becomes NaN |
| `keep_best` / `patience` / `val_map_every` | `true` / `0` / `1` | Best-epoch selection and early stopping on the validation mask mAP |
| `score_threshold` | `0.5` | Minimum score of the masks sent as evaluations and drawn in the overlays |
| `metrics_score_threshold` | `0.05` | Minimum score of the instances used for the COCO metrics |

## Export and Limits

- Only the `pytorch` export exists (the checkpoint itself). Mask2Former's dynamic post-processing does not trace reliably to TorchScript / ONNX. Load the checkpoint as described in `model-metadata`: the Mask2Former config is stored with the weights, so no network access is needed.
- Images are stretched to a square, so very elongated images lose their aspect ratio (masks are resized back to the original size for metrics and evaluations).
- Polygon annotations inside a `MASK` dataset are skipped; only RLE masks are used.
- The Hugging Face checkpoints are published under their own license: check it for your use case.

## Local Development

```bash
cd pipelines/mask2former
uv sync --python 3.14 --project training
```

The environment is pinned to Python 3.14 and `torch==2.10.0` (CUDA 12.8 wheels on Linux) to match the `picsellia/cuda:12.6.3-cudnn-ubuntu24.04-python3.14` Docker image.

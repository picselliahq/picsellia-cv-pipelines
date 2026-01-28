from __future__ import annotations

import os

import torch
from picsellia.types.enums import InferenceType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.core.models import PicselliaPolygonPrediction
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl
from torch.utils.data import DataLoader
from utils.steps_utils import (
    PicselliaLogger,
    build_datasets,
    build_label_maps,
    collate_fn,
    evaluate_one_epoch,
    get_maskrcnn_model,
    run_inference_on_asset,
    save_and_upload_artifacts,
    train_one_epoch,
)


def _load_transfer_learning_checkpoint(experiment, num_classes: int, backbone: str):
    """Load checkpoint from experiment for transfer learning.

    Args:
        experiment: Picsellia experiment object.
        num_classes: Number of classes for the new model.
        backbone: Backbone architecture.

    Returns:
        Loaded model with weights from checkpoint, or None if not available.
    """
    try:
        artifacts = experiment.list_artifacts()
        checkpoint_artifact = None
        for artifact in artifacts:
            if artifact.name == "checkpoint-latest":
                checkpoint_artifact = artifact
                break

        if checkpoint_artifact is None:
            print("  - No 'checkpoint-latest' artifact found in experiment")
            return None

        # Download the checkpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_artifact.download(tmp_dir)
            checkpoint_path = os.path.join(tmp_dir, checkpoint_artifact.filename)

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_num_classes = checkpoint.get("num_classes")
            checkpoint_backbone = checkpoint.get("backbone", "resnet50")

            print(
                f"  - Checkpoint classes: {checkpoint_num_classes}, current: {num_classes}"
            )
            print(f"  - Checkpoint backbone: {checkpoint_backbone}")

            if checkpoint_backbone != backbone:
                print(f"  - WARNING: Backbone mismatch, cannot use transfer learning")
                return None

            # Create model with checkpoint's num_classes first to load weights
            model = get_maskrcnn_model(
                num_classes=checkpoint_num_classes,
                backbone=backbone,
                pretrained=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            # If num_classes differs, replace the head layers
            if checkpoint_num_classes != num_classes:
                print(
                    f"  - Replacing classification heads for {num_classes} classes..."
                )
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(
                    in_features, num_classes
                )

                in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                model.roi_heads.mask_predictor = MaskRCNNPredictor(
                    in_features_mask, hidden_layer, num_classes
                )

            return model

    except Exception as e:
        print(f"  - Failed to load transfer learning checkpoint: {e}")
        return None


@step()
def train(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Train Mask R-CNN on COCO datasets and upload weights."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters

    print("=" * 60)
    print("MASK R-CNN TRAINING PIPELINE")
    print("=" * 60)

    print("\n[1/6] Building label maps...")
    id2label, label2id = build_label_maps(ds=picsellia_datasets["train"])
    num_classes = len(id2label) + 1
    print(f"  - Number of classes (including background): {num_classes}")
    print(f"  - Labels: {list(id2label.values())}")

    print("\n[2/6] Loading Mask R-CNN model...")
    print(f"  - Backbone: {hp.backbone}")
    print(f"  - Transfer learning enabled: {hp.transfer_learning}")

    model = None
    using_transfer_learning = False
    if hp.transfer_learning:
        print("  - Attempting to load checkpoint for transfer learning...")
        model = _load_transfer_learning_checkpoint(
            ctx.experiment, num_classes, hp.backbone
        )
        if model is not None:
            using_transfer_learning = True
            print("  - Transfer learning checkpoint loaded successfully")
        else:
            print(
                "  - Transfer learning checkpoint not available, falling back to pretrained weights"
            )

    if model is None:
        print("  - Loading pretrained ImageNet weights...")
        model = get_maskrcnn_model(
            num_classes=num_classes,
            backbone=hp.backbone,
            pretrained=True,
        )

    if using_transfer_learning:
        print("  - MODE: Transfer learning from previous checkpoint")
    else:
        print("  - MODE: Training from ImageNet pretrained weights")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device}")
    if device.type == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    model.to(device)
    print("  - Model loaded successfully")

    print("\n[3/6] Preparing datasets...")
    train_ds, val_ds = build_datasets(picsellia_datasets, label2id)
    print(f"  - Training samples: {len(train_ds)}")
    print(f"  - Validation samples: {len(val_ds)}")

    workers = int(os.getenv("WORKERS", "4"))
    print(f"  - DataLoader workers: {workers}")
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(hp.batch_size, 4),
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"  - Training batches per epoch: {len(train_loader)}")
    print(f"  - Validation batches per epoch: {len(val_loader)}")

    print("\n[4/6] Configuring optimizer and scheduler...")
    print(f"  - Optimizer: SGD")
    print(f"  - Learning rate: {hp.learning_rate}")
    print(f"  - Momentum: {hp.momentum}")
    print(f"  - Weight decay: {hp.weight_decay}")
    print(
        f"  - LR scheduler: StepLR (step_size={hp.lr_scheduler_step_size}, gamma={hp.lr_scheduler_gamma})"
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=hp.learning_rate,
        momentum=hp.momentum,
        weight_decay=hp.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hp.lr_scheduler_step_size,
        gamma=hp.lr_scheduler_gamma,
    )

    logger = PicselliaLogger(ctx.experiment)

    ctx.experiment.log_parameters(hp.to_dict())

    print("\n[5/6] Starting training...")
    print(f"  - Epochs: {hp.epochs}")
    print(f"  - Batch size: {hp.batch_size}")
    print(f"  - Image size: {hp.image_size}")
    print("-" * 60)

    for epoch in range(hp.epochs):
        print(f"\nEpoch {epoch + 1}/{hp.epochs}")
        print("-" * 40)

        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch, logger
        )
        print(f"  [Train] Loss: {train_metrics['loss']:.4f}")
        print(f"          - Classifier: {train_metrics['loss_classifier']:.4f}")
        print(f"          - Box Reg: {train_metrics['loss_box_reg']:.4f}")
        print(f"          - Mask: {train_metrics['loss_mask']:.4f}")
        print(f"          - Objectness: {train_metrics['loss_objectness']:.4f}")
        print(f"          - RPN Box Reg: {train_metrics['loss_rpn_box_reg']:.4f}")

        val_metrics = evaluate_one_epoch(model, val_loader, device, epoch, logger)
        print(f"  [Val]   Loss: {val_metrics['eval_loss']:.4f}")
        print(f"          - Classifier: {val_metrics['eval_loss_classifier']:.4f}")
        print(f"          - Box Reg: {val_metrics['eval_loss_box_reg']:.4f}")
        print(f"          - Mask: {val_metrics['eval_loss_mask']:.4f}")

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_metrics({"learning_rate": current_lr}, step=epoch)
        print(f"  [LR]    {current_lr:.6f}")

    print("\n" + "=" * 60)
    print("[6/6] Saving and uploading model artifacts...")
    save_and_upload_artifacts(
        picsellia_model,
        ctx.experiment,
        model,
        id2label,
        image_size=hp.image_size,
        backbone=hp.backbone,
    )
    print("  - Model saved and uploaded successfully")
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)


@step()
def evaluate(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Run inference on test split and evaluate in Picsellia."""
    ctx = Pipeline.get_active_context()

    print("\n" + "=" * 60)
    print("MASK R-CNN EVALUATION PIPELINE")
    print("=" * 60)

    print("\n[1/4] Loading trained model...")
    final_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name, "final")
    checkpoint_path = os.path.join(final_dir, "model.pth")
    print(f"  - Checkpoint path: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_classes = checkpoint["num_classes"]
    id2label = checkpoint["id2label"]
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Labels: {list(id2label.values())}")

    print(f"\n[2/4] Initializing model...")
    print(f"  - Backbone: {ctx.hyperparameters.backbone}")
    model = get_maskrcnn_model(
        num_classes=num_classes,
        backbone=ctx.hyperparameters.backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Device: {device}")
    if device.type == "cuda":
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    model.to(device)
    model.eval()
    print("  - Model loaded and set to eval mode")

    print("\n[3/4] Running inference on test set...")
    ds = picsellia_datasets["test"]
    total_assets = len(ds.assets)
    print(f"  - Test set size: {total_assets} images")
    print(f"  - Confidence threshold: 0.5")
    print(f"  - Mask threshold: 0.5")

    import time

    predictions: list[PicselliaPolygonPrediction] = []
    assets_with_predictions = 0
    total_polygons = 0
    inference_times: list[float] = []

    for i, asset in enumerate(ds.assets):
        if (i + 1) % 10 == 0 or (i + 1) == total_assets:
            print(f"  - Processing: {i + 1}/{total_assets} images...")

        start_time = time.perf_counter()
        pred = run_inference_on_asset(
            ds,
            asset,
            model,
            device,
            id2label,
            conf_thresh=0.5,
            mask_thresh=0.5,
        )
        inference_times.append(time.perf_counter() - start_time)

        if pred:
            predictions.append(pred)
            assets_with_predictions += 1
            total_polygons += len(pred.polygons)

    mean_inference_time = (
        sum(inference_times) / len(inference_times) if inference_times else 0.0
    )
    mean_inference_time_ms = mean_inference_time * 1000

    print(f"\n  Inference complete:")
    print(f"  - Images with predictions: {assets_with_predictions}/{total_assets}")
    print(f"  - Total polygons detected: {total_polygons}")
    if assets_with_predictions > 0:
        print(
            f"  - Average polygons per image: {total_polygons / assets_with_predictions:.1f}"
        )
    print(f"  - Mean inference time: {mean_inference_time_ms:.2f} ms")

    from picsellia.types.enums import LogType

    ctx.experiment.log(
        name="mean_inference_time_ms", data=mean_inference_time_ms, type=LogType.VALUE
    )

    print("\n[4/4] Evaluating predictions...")
    if predictions:
        training_labelmap = ctx.experiment.get_log("labelmap").data
        print(
            f"  - Sending {len(predictions)} predictions to Picsellia for evaluation..."
        )
        evaluate_model_impl(
            context=ctx,
            picsellia_predictions=predictions,
            inference_type=InferenceType.SEGMENTATION,
            assets=ds.assets,
            output_dir=os.path.join(ctx.working_dir, "evaluation"),
            training_labelmap=training_labelmap,
        )
        print("  - Evaluation metrics computed and logged")
    else:
        print("  - WARNING: No predictions to evaluate!")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

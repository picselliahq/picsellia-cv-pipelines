from __future__ import annotations

import os
import tempfile
import time

import torch
from picsellia import Asset
from picsellia.types.enums import InferenceType, LogType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from torch.utils.data import DataLoader
from utils.evaluation import (
    evaluate_detections,
    push_mask_evaluations,
    save_prediction_overlays,
    write_evaluation_files,
)
from utils.log import fmt_duration, get_logger
from utils.steps_utils import (
    PicselliaLogger,
    build_datasets,
    build_label_maps,
    collate_fn,
    compute_val_map,
    evaluate_one_epoch,
    get_maskrcnn_model,
    gpu_memory_gb,
    run_inference_on_asset,
    save_and_upload_artifacts,
    train_one_epoch,
)

log = get_logger()

# Upload the raw predictions as an artifact only below this size.
_MAX_PREDICTIONS_UPLOAD_BYTES = 50 * 1024 * 1024


def _load_transfer_learning_checkpoint(
    experiment,
    num_classes: int,
    backbone: str,
    image_size: int,
    trainable_backbone_layers: int,
):
    """Load checkpoint from experiment for transfer learning.

    Args:
        experiment: Picsellia experiment object.
        num_classes: Number of classes for the new model (with background).
        backbone: Backbone architecture.
        image_size: Shorter side images are resized to.
        trainable_backbone_layers: Number of backbone stages that are trained.

    Returns:
        Loaded model with weights from checkpoint, or None if not available.
    """
    try:
        checkpoint_artifact = next(
            (a for a in experiment.list_artifacts() if a.name == "checkpoint-latest"),
            None,
        )
        if checkpoint_artifact is None:
            log.warning("No 'checkpoint-latest' artifact found in the experiment")
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_artifact.download(tmp_dir)
            checkpoint_path = os.path.join(tmp_dir, checkpoint_artifact.filename)

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_num_classes = checkpoint.get("num_classes")
            checkpoint_backbone = checkpoint.get("backbone", "resnet50")

            log.info(
                "Checkpoint: %s classes (with background), backbone %s | current: %s classes, backbone %s",
                checkpoint_num_classes,
                checkpoint_backbone,
                num_classes,
                backbone,
            )

            if checkpoint_backbone != backbone:
                log.warning(
                    "Backbone mismatch (%s vs %s): transfer learning is not possible",
                    checkpoint_backbone,
                    backbone,
                )
                return None

            # Create model with checkpoint's num_classes first to load weights
            model = get_maskrcnn_model(
                num_classes=checkpoint_num_classes,
                backbone=backbone,
                pretrained=False,
                image_size=image_size,
                trainable_backbone_layers=trainable_backbone_layers,
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            # If num_classes differs, replace the head layers
            if checkpoint_num_classes != num_classes:
                log.info("Replacing the box and mask heads for %d classes", num_classes)
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

    except Exception:
        log.warning("Could not load the transfer learning checkpoint", exc_info=True)
        return None


def _init_model(ctx, hp, num_classes: int) -> torch.nn.Module:
    """Build the model to train: previous checkpoint if asked and possible, else COCO weights."""
    model = None
    if hp.transfer_learning:
        log.info("Transfer learning requested: loading the experiment checkpoint")
        model = _load_transfer_learning_checkpoint(
            ctx.experiment,
            num_classes,
            hp.backbone,
            hp.image_size,
            hp.trainable_backbone_layers,
        )
        if model is None:
            log.warning("Falling back to the torchvision COCO pretrained weights")

    if model is not None:
        log.info("Starting from the previous checkpoint (transfer learning)")
        return model

    log.info(
        "Starting from the torchvision COCO pretrained weights (heads re-initialized)"
    )
    return get_maskrcnn_model(
        num_classes=num_classes,
        backbone=hp.backbone,
        pretrained=True,
        image_size=hp.image_size,
        trainable_backbone_layers=hp.trainable_backbone_layers,
    )


def _log_data_stats(experiment, train_ds, val_ds) -> None:
    """Log the size of the train/val splits, in the console and as experiment values."""
    n_train_ann = len(train_ds.coco.getAnnIds(imgIds=train_ds.img_ids))
    n_val_ann = len(val_ds.coco.getAnnIds(imgIds=val_ds.img_ids))
    log.info(
        "Data: %d train images (%d masks), %d val images (%d masks)",
        len(train_ds),
        n_train_ann,
        len(val_ds),
        n_val_ann,
    )
    for name, value in {
        "dataset/train_images": len(train_ds),
        "dataset/val_images": len(val_ds),
        "dataset/train_masks": n_train_ann,
        "dataset/val_masks": n_val_ann,
    }.items():
        experiment.log(name=name, data=value, type=LogType.VALUE)


def _should_compute_val_map(hp, epoch: int) -> bool:
    """Validation mAP runs every ``val_map_every`` epochs, and always on the last one."""
    if hp.val_map_every <= 0:
        return False
    return (epoch + 1) % hp.val_map_every == 0 or epoch + 1 == hp.epochs


def _log_epoch_metrics(
    logger: PicselliaLogger,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    val_map: dict[str, float],
    learning_rate: float,
    epoch_seconds: float,
) -> None:
    """Send the metrics of one epoch to Picsellia (train/*, val/*)."""
    system_metrics = {"lr": learning_rate, "epoch_time_s": epoch_seconds}
    memory = gpu_memory_gb()
    if memory is not None:
        system_metrics["gpu_memory_peak_gb"] = memory
    logger.log_metrics(train_metrics, prefix="train")
    logger.log_metrics(system_metrics, prefix="train")
    logger.log_metrics(val_metrics, prefix="val")
    logger.log_metrics(
        {
            name: val_map[name]
            for name in ("mAP50-95(M)", "mAP50(M)", "mAP50-95(B)")
            if name in val_map
        },
        prefix="val",
    )


@step()
def train(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Train Mask R-CNN on COCO datasets and upload weights."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters

    export_format = ctx.export_parameters.export_format
    if export_format != "torchscript":
        raise ValueError(
            f"export_format '{export_format}' is not supported by this pipeline: "
            "only 'torchscript' is exported."
        )

    log.info(
        "Mask R-CNN training | backbone=%s image_size=%d epochs=%d batch_size=%d lr=%g",
        hp.backbone,
        hp.image_size,
        hp.epochs,
        hp.batch_size,
        hp.learning_rate,
    )

    id2label, label2id = build_label_maps(ds=picsellia_datasets["train"])
    num_classes = len(id2label) + 1
    log.info("Classes (%d + background): %s", len(id2label), list(id2label.values()))

    model = _init_model(ctx, hp, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(
            "Device: cuda (%s, %.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        log.warning("No GPU found: training on CPU will be very slow")
    model.to(device)

    train_ds, val_ds = build_datasets(picsellia_datasets, label2id)
    if len(train_ds) == 0:
        raise ValueError("The training split has no usable image (check the dataset).")
    _log_data_stats(ctx.experiment, train_ds, val_ds)

    workers = int(os.getenv("WORKERS", "4"))
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
    log.info(
        "DataLoaders: %d workers, %d train batches, %d val batches",
        workers,
        len(train_loader),
        len(val_loader),
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
    log.info(
        "Optimizer SGD (lr=%g momentum=%g weight_decay=%g) | StepLR (step=%d gamma=%g)",
        hp.learning_rate,
        hp.momentum,
        hp.weight_decay,
        hp.lr_scheduler_step_size,
        hp.lr_scheduler_gamma,
    )

    logger = PicselliaLogger(ctx.experiment)
    ctx.experiment.log_parameters(hp.to_dict())

    training_start = time.perf_counter()
    for epoch in range(hp.epochs):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            hp.epochs,
            log_every=hp.log_every,
        )
        val_metrics = evaluate_one_epoch(model, val_loader, device)

        val_map: dict[str, float] = {}
        if _should_compute_val_map(hp, epoch) and len(val_ds) > 0:
            val_map = compute_val_map(
                model,
                val_loader,
                val_ds,
                picsellia_datasets["val"].coco_file_path,
                device,
                conf_thresh=hp.metrics_score_threshold,
                mask_thresh=hp.mask_threshold,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        _log_epoch_metrics(
            logger, train_metrics, val_metrics, val_map, current_lr, epoch_seconds
        )

        log.info(
            "epoch %d/%d done in %s | train loss %.4f%s%s | lr %.2e",
            epoch + 1,
            hp.epochs,
            fmt_duration(epoch_seconds),
            train_metrics["loss"],
            f" | val loss {val_metrics['loss']:.4f}" if val_metrics else "",
            f" | val mAP50-95(M) {val_map['mAP50-95(M)']:.4f}" if val_map else "",
            current_lr,
        )

    total_seconds = time.perf_counter() - training_start
    ctx.experiment.log(
        name="train/total_time_s", data=total_seconds, type=LogType.VALUE
    )
    log.info("Training finished in %s, saving the model", fmt_duration(total_seconds))

    sample_source = val_ds if len(val_ds) > 0 else train_ds
    sample_images = [sample_source[i][0] for i in range(min(5, len(sample_source)))]
    save_and_upload_artifacts(
        picsellia_model,
        ctx.experiment,
        model,
        id2label,
        image_size=hp.image_size,
        backbone=hp.backbone,
        sample_images=sample_images,
        score_threshold=hp.score_threshold,
        mask_threshold=hp.mask_threshold,
    )
    log.info("Training step completed")


@step()
def evaluate(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Evaluate on the test split: COCO metrics, Picsellia evaluations and samples."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters

    final_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name, "final")
    checkpoint_path = os.path.join(final_dir, "model.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_classes = checkpoint["num_classes"]
    id2label = checkpoint["id2label"]
    backbone = checkpoint.get("backbone", hp.backbone)
    image_size = checkpoint.get("image_size", hp.image_size)
    log.info(
        "Evaluating %s | %d classes | backbone=%s image_size=%d",
        checkpoint_path,
        num_classes - 1,
        backbone,
        image_size,
    )

    model = get_maskrcnn_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
        image_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ds = picsellia_datasets["test"]
    total_assets = len(ds.assets)
    log.info(
        "Test set: %d images | metrics at score >= %g, Picsellia evaluations at score >= %g, mask threshold %g",
        total_assets,
        hp.metrics_score_threshold,
        hp.score_threshold,
        hp.mask_threshold,
    )

    # Ground-truth ids: assets are matched to COCO images by file name, and labels
    # to COCO categories by name.
    file_name_to_image_id = {
        img["file_name"]: img["id"] for img in ds.coco_data["images"]
    }
    name2cat_id = {cat["name"]: cat["id"] for cat in ds.coco_data["categories"]}
    cat_id_to_name = {cat_id: name for name, cat_id in name2cat_id.items()}
    # The model predicts 1-based labels (0 is the background)
    label_to_cat_id = {
        index + 1: name2cat_id[name]
        for index, name in id2label.items()
        if name in name2cat_id
    }

    detections: list[dict] = []
    evaluated_image_ids: list[int] = []
    image_id_to_asset: dict[int, Asset] = {}
    image_id_to_path: dict[int, str] = {}
    forward_times: list[float] = []
    total_times: list[float] = []
    log_every = hp.log_every if hp.log_every > 0 else max(1, total_assets // 10)
    inference_start = time.perf_counter()

    for i, asset in enumerate(ds.assets, start=1):
        image_id = file_name_to_image_id.get(asset.id_with_extension)
        if image_id is None:
            log.warning("%s is not in the COCO file, skipping", asset.id_with_extension)
            continue

        asset_start = time.perf_counter()
        asset_detections, forward_seconds = run_inference_on_asset(
            ds,
            asset,
            model,
            device,
            label_to_cat_id,
            image_id=image_id,
            conf_thresh=hp.metrics_score_threshold,
            mask_thresh=hp.mask_threshold,
        )
        total_times.append(time.perf_counter() - asset_start)
        forward_times.append(forward_seconds)

        evaluated_image_ids.append(image_id)
        image_id_to_asset[image_id] = asset
        image_id_to_path[image_id] = os.path.join(
            ds.images_dir, asset.id_with_extension
        )
        detections.extend(asset_detections)

        if i % log_every == 0 or i == total_assets:
            elapsed = time.perf_counter() - inference_start
            log.info(
                "inference %d/%d images | %d masks | ETA %s",
                i,
                total_assets,
                len(detections),
                fmt_duration((total_assets - i) * elapsed / i),
            )

    # The first forward pays the CUDA warm-up: leave it out of the mean.
    steady = forward_times[1:] or forward_times
    forward_ms = 1000 * sum(steady) / len(steady) if steady else 0.0
    total_ms = 1000 * sum(total_times) / len(total_times) if total_times else 0.0
    images_with_masks = len({d["image_id"] for d in detections})
    log.info(
        "Inference done: %d masks on %d/%d images | forward %.1f ms, %.1f ms per image with loading and encoding",
        len(detections),
        images_with_masks,
        len(evaluated_image_ids),
        forward_ms,
        total_ms,
    )

    # 1. COCO metrics on every detection above the low metrics threshold.
    result = evaluate_detections(
        gt_coco_path=ds.coco_file_path,
        detections=detections,
        image_ids=evaluated_image_ids,
    )
    output_dir = os.path.join(ctx.working_dir, "evaluation")
    files = write_evaluation_files(output_dir, detections, result)

    experiment = ctx.experiment
    for name, value in result["values"].items():
        experiment.log(name=f"test/{name}", data=round(value, 4), type=LogType.VALUE)
    experiment.log(name="test/metrics", data=result["table"], type=LogType.TABLE)
    for name, value in {
        "test/inference_time_ms": forward_ms,
        "test/time_per_image_ms": total_ms,
        "test/images_evaluated": len(evaluated_image_ids),
        "test/masks_detected": len(detections),
    }.items():
        experiment.log(name=name, data=round(value, 2), type=LogType.VALUE)
    log.info(
        "Test metrics: mAP50-95(M) %.4f | mAP50(M) %.4f | mAR100(M) %.4f | mAP50-95(B) %.4f",
        result["values"]["mAP50-95(M)"],
        result["values"]["mAP50(M)"],
        result["values"]["mAR100(M)"],
        result["values"]["mAP50-95(B)"],
    )

    # 2. What is sent to Picsellia is filtered with the (higher) score threshold.
    shown = [d for d in detections if d["score"] >= hp.score_threshold]
    log.info(
        "%d of the %d masks have a score >= %g",
        len(shown),
        len(detections),
        hp.score_threshold,
    )

    if shown:
        n_pushed = push_mask_evaluations(
            experiment=experiment,
            ds=ds,
            detections=shown,
            image_id_to_asset=image_id_to_asset,
            cat_id_to_name=cat_id_to_name,
            inference_type=InferenceType.MASK,
        )
        log.info(
            "%d evaluations added to the experiment, platform metrics computed",
            n_pushed,
        )
    else:
        log.warning(
            "No mask above the score threshold: no evaluation sent to Picsellia"
        )

    # 3. A few predictions drawn on the images, then the evaluation files.
    overlays = save_prediction_overlays(
        image_id_to_path,
        shown,
        cat_id_to_name,
        os.path.join(output_dir, "overlays"),
        max_images=hp.overlay_images,
    )
    for n, overlay_path in enumerate(overlays, start=1):
        experiment.log(
            name=f"test/prediction_{n:02d}", data=overlay_path, type=LogType.IMAGE
        )
    log.info("%d prediction overlays logged", len(overlays))

    experiment.store("mask-metrics", files["metrics"])
    if os.path.getsize(files["predictions"]) <= _MAX_PREDICTIONS_UPLOAD_BYTES:
        experiment.store("mask-predictions", files["predictions"])
    else:
        log.warning(
            "mask_predictions.json is larger than %d MB: not uploaded (kept in %s)",
            _MAX_PREDICTIONS_UPLOAD_BYTES // (1024 * 1024),
            files["predictions"],
        )
    log.info("Evaluation step completed")

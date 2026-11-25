"""Helpers of the ``train`` step: checks, model initialisation, loaders, reporting, best checkpoint."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any

import torch
from picsellia.types.enums import LogType
from torch.utils.data import DataLoader
from utils.log import get_logger
from utils.steps_utils import (
    CocoMaskDataset,
    PicselliaLogger,
    collate_fn,
    get_maskrcnn_model,
    gpu_memory_gb,
    replace_predictors,
    seed_worker,
    snapshot_state,
)

log = get_logger()

LR_SCHEDULERS = ("cosine", "step")


def validate_hyperparameters(hp: Any, export_format: str) -> None:
    """Fail at the start, with a clear message, on a value the pipeline cannot honour."""
    if export_format != "torchscript":
        raise ValueError(
            f"export_format '{export_format}' is not supported by this pipeline: "
            "only 'torchscript' is exported."
        )
    if hp.lr_scheduler not in LR_SCHEDULERS:
        raise ValueError(
            f"lr_scheduler '{hp.lr_scheduler}' is unknown: use one of {LR_SCHEDULERS}."
        )
    if not 0 <= hp.trainable_backbone_layers <= 5:
        raise ValueError("trainable_backbone_layers must be between 0 and 5.")
    if not 0.0 <= hp.hflip_prob <= 1.0:
        raise ValueError("hflip_prob must be between 0 and 1.")
    if not 0.0 <= hp.scale_jitter < 1.0:
        raise ValueError("scale_jitter must be in [0, 1).")


def log_device(device: torch.device) -> None:
    """Log the device the training runs on."""
    if device.type == "cuda":
        log.info(
            "Device: cuda (%s, %.1f GB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        log.warning("No GPU found: training on CPU will be very slow")


def _resolve_checkpoint(
    picsellia_model: Any, experiment: Any, tmp_dir: str
) -> str | None:
    """Path of the checkpoint to start from.

    The weights ``build_model`` already downloaded from the model version are
    preferred; the ``checkpoint-latest`` artifact of the experiment is the fallback.
    """
    path = getattr(picsellia_model, "pretrained_weights_path", None)
    if path and os.path.isfile(path):
        log.info("Checkpoint: model version weights (%s)", path)
        return path
    artifact = next(
        (a for a in experiment.list_artifacts() if a.name == "checkpoint-latest"),
        None,
    )
    if artifact is None:
        log.warning("No checkpoint-latest in the model version nor in the experiment")
        return None
    artifact.download(tmp_dir)
    log.info("Checkpoint: experiment artifact checkpoint-latest")
    return os.path.join(tmp_dir, artifact.filename)


def load_transfer_learning_checkpoint(
    picsellia_model: Any, experiment: Any, num_classes: int, hp: Any
) -> torch.nn.Module | None:
    """Model initialised from a previous checkpoint, or None when that is not possible.

    The box and mask predictors are replaced when the number of classes differs.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = _resolve_checkpoint(picsellia_model, experiment, tmp_dir)
            if checkpoint_path is None:
                return None
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            checkpoint_num_classes = checkpoint.get("num_classes")
            checkpoint_backbone = checkpoint.get("backbone", "resnet50")
            log.info(
                "Checkpoint: %s classes (with background), backbone %s | current: %s classes, backbone %s",
                checkpoint_num_classes,
                checkpoint_backbone,
                num_classes,
                hp.backbone,
            )
            if checkpoint_backbone != hp.backbone:
                log.warning(
                    "Backbone mismatch (%s vs %s): transfer learning is not possible",
                    checkpoint_backbone,
                    hp.backbone,
                )
                return None

            # Same architecture as the checkpoint first, to load its weights
            model = get_maskrcnn_model(
                num_classes=checkpoint_num_classes,
                backbone=hp.backbone,
                pretrained=False,
                image_size=hp.image_size,
                trainable_backbone_layers=hp.trainable_backbone_layers,
                scale_jitter=hp.scale_jitter,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint_num_classes != num_classes:
                log.info("Replacing the box and mask heads for %d classes", num_classes)
                replace_predictors(model, num_classes)
            return model
    except Exception:
        log.warning("Could not load the transfer learning checkpoint", exc_info=True)
        return None


def init_model(
    picsellia_model: Any, experiment: Any, hp: Any, num_classes: int
) -> torch.nn.Module:
    """Model to train: the previous checkpoint if asked and possible, else COCO weights."""
    if hp.transfer_learning:
        log.info("Transfer learning requested")
        model = load_transfer_learning_checkpoint(
            picsellia_model, experiment, num_classes, hp
        )
        if model is not None:
            log.info("Starting from the checkpoint (transfer learning)")
            return model
        log.warning("Falling back to the torchvision COCO pretrained weights")

    log.info(
        "Starting from the torchvision COCO pretrained weights (heads re-initialized)"
    )
    return get_maskrcnn_model(
        num_classes=num_classes,
        backbone=hp.backbone,
        pretrained=True,
        image_size=hp.image_size,
        trainable_backbone_layers=hp.trainable_backbone_layers,
        scale_jitter=hp.scale_jitter,
    )


def log_data_stats(
    experiment: Any, train_ds: CocoMaskDataset, val_ds: CocoMaskDataset
) -> None:
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


def make_loaders(
    hp: Any, train_ds: CocoMaskDataset, val_ds: CocoMaskDataset
) -> tuple[DataLoader, DataLoader]:
    """Train (shuffled, seeded) and validation loaders; workers come from ``WORKERS``."""
    workers = int(os.getenv("WORKERS", "4"))
    generator = torch.Generator()
    generator.manual_seed(hp.seed)
    common: dict[str, Any] = {
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(hp.batch_size, 4), shuffle=False, **common
    )
    log.info(
        "DataLoaders: %d workers, %d train batches, %d val batches",
        workers,
        len(train_loader),
        len(val_loader),
    )
    return train_loader, val_loader


def should_compute_val_map(hp: Any, epoch: int) -> bool:
    """Validation mAP runs every ``val_map_every`` epochs, and always on the last one."""
    if hp.val_map_every <= 0:
        return False
    return (epoch + 1) % hp.val_map_every == 0 or epoch + 1 == hp.epochs


def epoch_score(
    hp: Any, val_metrics: dict[str, float], val_map: dict[str, float]
) -> float | None:
    """Score used to pick the best epoch (higher is better), None when not evaluated.

    The validation mask mAP when it is computed, else the negative validation loss
    when the mAP is disabled (``val_map_every = 0``).
    """
    if val_map:
        return val_map["mAP50-95(M)"]
    if hp.val_map_every <= 0 and val_metrics:
        return -val_metrics["loss"]
    return None


class BestCheckpoint:
    """Keeps the best weights seen so far and tells when to stop early.

    Only epochs that have a score count: with ``val_map_every = 3``, early stopping
    patience is counted in evaluated epochs. Ties go to the most recent epoch.
    """

    def __init__(self, patience: int = 0) -> None:
        self.patience = patience
        self.score = -math.inf
        self.epoch = -1
        self.state: dict[str, torch.Tensor] | None = None
        self.stale_evaluations = 0

    def update(self, epoch: int, score: float | None, model: torch.nn.Module) -> None:
        if score is None:
            return
        if score >= self.score:
            self.score, self.epoch = score, epoch
            self.state = snapshot_state(model)
            self.stale_evaluations = 0
        else:
            self.stale_evaluations += 1

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.stale_evaluations >= self.patience

    def restore(self, model: torch.nn.Module) -> bool:
        """Load the best weights into ``model``; False when there is none."""
        if self.state is None:
            return False
        model.load_state_dict(self.state)
        return True


def log_epoch_metrics(
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

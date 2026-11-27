"""Training helpers: checks, loaders, optimisation loops, validation, best checkpoint, logs."""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any

import numpy as np
import torch
from picsellia import Experiment
from picsellia.types.enums import InferenceType, LogType
from torch.utils.data import DataLoader
from utils.dataset import Mask2FormerDataset, collate_fn
from utils.evaluation import evaluate_detections
from utils.log import fmt_duration, get_logger
from utils.model import InstancePredictor, instances_to_detections, snapshot_state

log = get_logger()

LR_SCHEDULERS = ("cosine", "step")
EXPORT_FORMATS = ("pytorch",)


# --------------------------------------------------------------------------- checks
def validate_hyperparameters(hp: Any, export_format: str) -> None:
    """Fail at the start, with a clear message, on a value the pipeline cannot honour."""
    if export_format not in EXPORT_FORMATS:
        raise ValueError(
            f"export_format '{export_format}' is not supported by this pipeline: "
            f"only {EXPORT_FORMATS} is exported."
        )
    if hp.lr_scheduler not in LR_SCHEDULERS:
        raise ValueError(
            f"lr_scheduler '{hp.lr_scheduler}' is unknown: use one of {LR_SCHEDULERS}."
        )
    if not 0.0 <= hp.hflip_prob <= 1.0:
        raise ValueError("hflip_prob must be between 0 and 1.")
    if hp.image_size < 64:
        raise ValueError("image_size must be at least 64.")
    if hp.batch_size < 1 or hp.eval_batch_size < 1:
        raise ValueError("batch_size and eval_batch_size must be at least 1.")


def validate_dataset_types(datasets: Any) -> None:
    """Fail at the start if a split does not come from a MASK dataset version.

    This pipeline reads RLE masks. Polygon (SEGMENTATION) datasets are handled by the
    ``maskrcnn/training`` pipeline, or can be converted with ``polygons_to_RLE_mask``.
    """
    for split in ("train", "val", "test"):
        version = datasets[split].dataset_version
        if version.type != InferenceType.MASK:
            raise ValueError(
                f"The '{split}' dataset version '{version.version}' is of type "
                f"{getattr(version.type, 'name', version.type)}, but this pipeline needs "
                "MASK dataset versions (RLE masks). For SEGMENTATION (polygon) datasets, "
                "use the maskrcnn/training pipeline, or convert them with "
                "polygons_to_RLE_mask."
            )


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


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - DataLoader signature
    """Seed python and numpy in a DataLoader worker from torch's per-worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def gpu_memory_gb() -> float | None:
    """Peak GPU memory allocated since the last reset, in GB (None without CUDA)."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9


# --------------------------------------------------------------------------- data
def log_data_stats(
    experiment: Any, train_ds: Mask2FormerDataset, val_ds: Mask2FormerDataset
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
    hp: Any, train_ds: Mask2FormerDataset, val_ds: Mask2FormerDataset
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
        val_ds, batch_size=hp.eval_batch_size, shuffle=False, **common
    )
    log.info(
        "DataLoaders: %d workers, %d train batches, %d val batches",
        workers,
        len(train_loader),
        len(val_loader),
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- schedule
def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    epochs: int,
    learning_rate: float,
    step_size: int,
    gamma: float,
):
    """Epoch-level learning rate scheduler: ``cosine`` (down to 1% of the lr) or ``step``."""
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.01
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    raise ValueError(f"Unknown lr_scheduler '{name}': use 'cosine' or 'step'.")


def use_bf16(amp: bool, device: torch.device) -> bool:
    """bfloat16 autocast needs a GPU that supports it (no loss scaling is needed)."""
    return bool(amp and device.type == "cuda" and torch.cuda.is_bf16_supported())


# --------------------------------------------------------------------------- loops
def _log_progress(
    epoch: int,
    epochs: int,
    iteration: int,
    n_batches: int,
    mean_loss: float,
    learning_rate: float,
    elapsed: float,
) -> None:
    """One progress line: loss, learning rate, speed and ETA."""
    rate = iteration / elapsed
    memory = gpu_memory_gb()
    log.info(
        "epoch %d/%d | iter %d/%d | loss %.4f | lr %.2e | %.2f it/s | ETA %s%s",
        epoch + 1,
        epochs,
        iteration,
        n_batches,
        mean_loss,
        learning_rate,
        rate,
        fmt_duration((n_batches - iteration) / rate),
        f" | gpu {memory:.1f} GB" if memory is not None else "",
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move a batch to the device: Mask2Former's forward arguments only."""
    return {
        "pixel_values": batch["pixel_values"].to(device),
        "mask_labels": [m.to(device) for m in batch["mask_labels"]],
        "class_labels": [c.to(device) for c in batch["class_labels"]],
    }


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    log_every: int = 0,
    clip_grad_norm: float = 0.0,
    warmup_iters: int = 0,
    bf16: bool = False,
) -> dict[str, float]:
    """Train for one epoch, logging progress with an ETA.

    Args:
        log_every: Log a progress line every N iterations (0 = about 10 per epoch).
        clip_grad_norm: Max gradient norm (0 disables clipping).
        warmup_iters: Linear learning rate warmup over the first iterations of the
            first epoch (from 0.1% of the learning rate).
        bf16: Train with bfloat16 autocast (GPU only).

    Returns:
        ``{"loss": average total loss}``.

    Raises:
        ValueError: If the data loader is empty.
        FloatingPointError: If the loss becomes NaN or infinite.
    """
    model.train()
    n_batches = len(data_loader)
    if n_batches == 0:
        raise ValueError("The training DataLoader is empty: no image to train on.")
    if log_every <= 0:
        log_every = max(1, n_batches // 10)

    warmup = None
    if epoch == 0 and warmup_iters > 0 and n_batches > 1:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, total_iters=min(warmup_iters, n_batches - 1)
        )

    total_loss = 0.0
    start = time.perf_counter()
    for iteration, batch in enumerate(data_loader, start=1):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=bf16
        ):
            loss = model(**_to_device(batch, device)).loss

        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f"Loss is {loss_value} at epoch {epoch + 1}, iteration {iteration}. "
                "Try a lower learning_rate."
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if warmup is not None:
            warmup.step()

        total_loss += loss_value
        if iteration % log_every == 0 or iteration == n_batches:
            _log_progress(
                epoch,
                epochs,
                iteration,
                n_batches,
                total_loss / iteration,
                optimizer.param_groups[-1]["lr"],
                time.perf_counter() - start,
            )

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module, data_loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Validation loss (Mask2Former returns it in eval mode when labels are given).

    Returns an empty dict when the validation loader is empty.
    """
    model.eval()
    n_batches = len(data_loader)
    if n_batches == 0:
        return {}
    total = 0.0
    for batch in data_loader:
        total += float(model(**_to_device(batch, device)).loss)
    return {"loss": total / n_batches}


@torch.no_grad()
def compute_val_map(
    predictor: InstancePredictor,
    data_loader: DataLoader,
    val_ds: Mask2FormerDataset,
    gt_coco_path: str,
    score_threshold: float,
) -> dict[str, float]:
    """COCO mask/box metrics on the validation split (model left in eval mode)."""
    class_to_cat_id = {cls: cat for cat, cls in val_ds.cat_to_class.items()}
    detections: list[dict[str, Any]] = []
    for batch in data_loader:
        instances = predictor.predict(
            batch["pixel_values"], batch["sizes"], score_threshold
        )
        for image_id, image_instances in zip(
            batch["image_ids"], instances, strict=True
        ):
            detections.extend(
                instances_to_detections(image_instances, image_id, class_to_cat_id)
            )
    return evaluate_detections(gt_coco_path, detections, val_ds.img_ids)["values"]


# --------------------------------------------------------------------------- selection
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


# --------------------------------------------------------------------------- logs
class PicselliaLogger:
    """Logger for training metrics to Picsellia."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def log_metrics(self, metrics: dict[str, float], prefix: str | None = None) -> None:
        """Log scalar metrics as line charts, named ``<prefix>/<metric>``.

        Non-finite values are skipped: they cannot be charted.
        """
        for name, value in metrics.items():
            if isinstance(value, int | float) and math.isfinite(value):
                log_name = f"{prefix}/{name}" if prefix else name
                self.experiment.log(name=log_name, data=float(value), type=LogType.LINE)


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

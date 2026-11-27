from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from picsellia import Asset
from picsellia.types.enums import InferenceType, LogType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from utils.dataset import Mask2FormerDataset, RandomHorizontalFlip
from utils.evaluation import (
    evaluate_detections,
    push_mask_evaluations,
    save_prediction_overlays,
    write_evaluation_files,
)
from utils.inference import run_inference_on_assets
from utils.log import fmt_duration, get_logger
from utils.model import (
    DEFAULT_HF_MODEL,
    PRETRAINED_WEIGHTS_NAME,
    InstancePredictor,
    build_label_maps,
    build_model_metadata,
    build_param_groups,
    checkpoint_dict,
    load_model_from_checkpoint,
    load_pretrained,
    resolve_pretrained_source,
)
from utils.training import (
    BestCheckpoint,
    PicselliaLogger,
    build_lr_scheduler,
    compute_val_map,
    epoch_score,
    evaluate_one_epoch,
    log_data_stats,
    log_device,
    log_epoch_metrics,
    make_loaders,
    seed_everything,
    should_compute_val_map,
    train_one_epoch,
    use_bf16,
    validate_dataset_types,
    validate_hyperparameters,
)

log = get_logger()

# Upload the raw predictions as an artifact only below this size.
_MAX_PREDICTIONS_UPLOAD_BYTES = 50 * 1024 * 1024


def _build_datasets(
    datasets: DatasetCollection[CocoDataset],
    label2id: dict[str, int],
    hp: Any,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
) -> tuple[Mask2FormerDataset, Mask2FormerDataset]:
    """Train (with horizontal flip) and validation datasets."""
    common = {
        "label2id": label2id,
        "image_size": hp.image_size,
        "image_mean": image_mean,
        "image_std": image_std,
    }
    train_ds = Mask2FormerDataset(
        images_dir=datasets["train"].images_dir,
        ann_json=datasets["train"].coco_file_path,
        transforms=RandomHorizontalFlip(hp.hflip_prob) if hp.hflip_prob > 0 else None,
        **common,
    )
    val_ds = Mask2FormerDataset(
        images_dir=datasets["val"].images_dir,
        ann_json=datasets["val"].coco_file_path,
        **common,
    )
    return train_ds, val_ds


def _compute_val_map(
    hp: Any,
    epoch: int,
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    val_ds: Mask2FormerDataset,
    gt_coco_path: str,
    device: torch.device,
) -> dict[str, float]:
    """Validation metrics of this epoch, or an empty dict when they are not due."""
    if not should_compute_val_map(hp, epoch) or len(val_ds) == 0:
        return {}
    predictor = InstancePredictor(
        model,
        device,
        hp.image_size,
        val_ds.image_mean,
        val_ds.image_std,
        mask_threshold=hp.mask_threshold,
        overlap_mask_area_threshold=hp.overlap_mask_area_threshold,
    )
    return compute_val_map(
        predictor, val_loader, val_ds, gt_coco_path, hp.metrics_score_threshold
    )


def _finalize_best(
    experiment: Any, hp: Any, best: BestCheckpoint, model: torch.nn.Module
) -> None:
    """Log the best epoch and, if asked, load its weights before the model is saved."""
    if best.epoch < 0:
        log.info("No validation score: the weights of the last epoch are kept")
        return
    experiment.log(name="train/best_epoch", data=best.epoch + 1, type=LogType.VALUE)
    experiment.log(
        name="train/best_score", data=round(best.score, 4), type=LogType.VALUE
    )
    if hp.keep_best and best.restore(model):
        log.info(
            "Best epoch: %d (score %.4f), its weights are the ones saved",
            best.epoch + 1,
            best.score,
        )
    else:
        log.info(
            "Best epoch: %d (score %.4f), the weights of the last epoch are saved",
            best.epoch + 1,
            best.score,
        )


def _save_and_upload_artifacts(
    picsellia_model: Model,
    experiment: Any,
    model: torch.nn.Module,
    id2label: dict[int, str],
    hp: Any,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    pretrained_source: str,
) -> None:
    """Write the checkpoint and its metadata, then upload them to the experiment."""
    final_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name, "final")
    os.makedirs(final_dir, exist_ok=True)

    checkpoint_path = os.path.join(final_dir, "model.pth")
    torch.save(
        checkpoint_dict(
            model, id2label, hp.image_size, image_mean, image_std, pretrained_source
        ),
        checkpoint_path,
    )
    # Stored through the experiment: `Model.save_artifact_to_experiment` has a different
    # signature depending on the engine version.
    experiment.store("checkpoint-latest", checkpoint_path)
    log.info("Uploaded artifact checkpoint-latest (%s)", checkpoint_path)

    metadata_path = os.path.join(final_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            build_model_metadata(
                id2label,
                hp.image_size,
                image_mean,
                image_std,
                pretrained_source,
                hp.score_threshold,
                hp.mask_threshold,
            ),
            f,
            indent=2,
        )
    experiment.store("model-metadata", metadata_path)
    log.info("Uploaded artifact model-metadata (%s)", metadata_path)


@step()
def train(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Fine-tune Mask2Former on COCO datasets with RLE masks and upload the weights."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters
    validate_hyperparameters(hp, ctx.export_parameters.export_format)
    validate_dataset_types(picsellia_datasets)
    seed_everything(hp.seed)

    pretrained_source = resolve_pretrained_source(
        picsellia_model,
        ctx.experiment,
        hp.pretrained_model,
        os.path.join(ctx.working_dir, "models"),
    )
    # What the checkpoint records: never a temporary download path.
    pretrained_label = (
        pretrained_source
        if hp.pretrained_model or pretrained_source == DEFAULT_HF_MODEL
        else f"model version file {PRETRAINED_WEIGHTS_NAME}"
    )
    log.info(
        "Mask2Former training | model=%s image_size=%d epochs=%d batch_size=%d lr=%g seed=%d",
        pretrained_label,
        hp.image_size,
        hp.epochs,
        hp.batch_size,
        hp.learning_rate,
        hp.seed,
    )
    id2label, label2id = build_label_maps(picsellia_datasets["train"].labelmap)
    log.info("Classes (%d): %s", len(id2label), list(id2label.values()))

    model, image_mean, image_std = load_pretrained(pretrained_source, id2label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_device(device)
    model.to(device)

    train_ds, val_ds = _build_datasets(
        picsellia_datasets, label2id, hp, image_mean, image_std
    )
    if len(train_ds) == 0:
        raise ValueError("The training split has no usable image (check the dataset).")
    log_data_stats(ctx.experiment, train_ds, val_ds)
    train_loader, val_loader = make_loaders(hp, train_ds, val_ds)

    optimizer = torch.optim.AdamW(
        build_param_groups(
            model, hp.learning_rate, hp.backbone_lr_multiplier, hp.weight_decay
        )
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        hp.lr_scheduler,
        hp.epochs,
        hp.learning_rate,
        hp.lr_scheduler_step_size,
        hp.lr_scheduler_gamma,
    )
    bf16 = use_bf16(hp.amp, device)
    log.info(
        "AdamW (lr=%g, backbone x%g, weight_decay=%g) | %s schedule, %d warmup iterations | bfloat16 %s | grad clip %g | hflip %g",
        hp.learning_rate,
        hp.backbone_lr_multiplier,
        hp.weight_decay,
        hp.lr_scheduler,
        hp.warmup_iters,
        "on" if bf16 else "off",
        hp.clip_grad_norm,
        hp.hflip_prob,
    )

    logger = PicselliaLogger(ctx.experiment)
    ctx.experiment.log_parameters(hp.to_dict())
    best = BestCheckpoint(patience=hp.patience)

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
            clip_grad_norm=hp.clip_grad_norm,
            warmup_iters=hp.warmup_iters,
            bf16=bf16,
        )
        val_metrics = evaluate_one_epoch(model, val_loader, device)
        val_map = _compute_val_map(
            hp,
            epoch,
            model,
            val_loader,
            val_ds,
            picsellia_datasets["val"].coco_file_path,
            device,
        )
        best.update(epoch, epoch_score(hp, val_metrics, val_map), model)

        current_lr = optimizer.param_groups[-1]["lr"]
        lr_scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        log_epoch_metrics(
            logger, train_metrics, val_metrics, val_map, current_lr, epoch_seconds
        )

        log.info(
            "epoch %d/%d done in %s | train loss %.4f%s%s | lr %.2e%s",
            epoch + 1,
            hp.epochs,
            fmt_duration(epoch_seconds),
            train_metrics["loss"],
            f" | val loss {val_metrics['loss']:.4f}" if val_metrics else "",
            f" | val mAP50-95(M) {val_map['mAP50-95(M)']:.4f}" if val_map else "",
            current_lr,
            " | best so far" if best.epoch == epoch else "",
        )
        if best.should_stop:
            log.info(
                "Early stopping: no improvement for %d evaluated epochs", hp.patience
            )
            break

    total_seconds = time.perf_counter() - training_start
    ctx.experiment.log(
        name="train/total_time_s", data=total_seconds, type=LogType.VALUE
    )
    log.info("Training finished in %s", fmt_duration(total_seconds))
    _finalize_best(ctx.experiment, hp, best, model)
    _save_and_upload_artifacts(
        picsellia_model,
        ctx.experiment,
        model,
        id2label,
        hp,
        image_mean,
        image_std,
        pretrained_label,
    )
    log.info("Training step completed")


@dataclass
class _InferenceResult:
    """What the test-set inference produced."""

    detections: list[dict] = field(default_factory=list)
    image_ids: list[int] = field(default_factory=list)
    assets: dict[int, Asset] = field(default_factory=dict)
    paths: dict[int, str] = field(default_factory=dict)
    predict_ms: float = 0.0
    total_ms: float = 0.0


def _run_test_inference(
    ds: Any,
    predictor: InstancePredictor,
    class_to_cat_id: dict[int, int],
    file_name_to_image_id: dict[str, int],
    hp: Any,
) -> _InferenceResult:
    """Run the model on the test split by batches, logging progress with an ETA."""
    result = _InferenceResult()
    items: list[tuple[Asset, int]] = []
    for asset in ds.assets:
        image_id = file_name_to_image_id.get(asset.id_with_extension)
        if image_id is None:
            log.warning("%s is not in the COCO file, skipping", asset.id_with_extension)
            continue
        items.append((asset, image_id))

    total = len(items)
    log_every = hp.log_every if hp.log_every > 0 else max(1, total // 10)
    predict_batches: list[tuple[float, int]] = []
    done = 0
    start = time.perf_counter()
    for batch_results, predict_seconds in run_inference_on_assets(
        ds.images_dir,
        items,
        predictor,
        class_to_cat_id,
        score_threshold=hp.metrics_score_threshold,
        batch_size=hp.eval_batch_size,
    ):
        for asset, image_id, detections in batch_results:
            result.detections.extend(detections)
            result.image_ids.append(image_id)
            result.assets[image_id] = asset
            result.paths[image_id] = os.path.join(
                ds.images_dir, asset.id_with_extension
            )
        previous, done = done, done + len(batch_results)
        predict_batches.append((predict_seconds, len(batch_results)))
        if done // log_every > previous // log_every or done == total:
            log.info(
                "inference %d/%d images | %d masks | ETA %s",
                done,
                total,
                len(result.detections),
                fmt_duration((total - done) * (time.perf_counter() - start) / done),
            )

    # The first batch pays the CUDA warm-up: leave it out of the mean.
    steady = predict_batches[1:] or predict_batches
    n_steady = sum(n for _, n in steady)
    result.predict_ms = 1000 * sum(t for t, _ in steady) / n_steady if n_steady else 0.0
    result.total_ms = 1000 * (time.perf_counter() - start) / done if done else 0.0
    return result


@step()
def evaluate(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Evaluate on the test split: COCO metrics, Picsellia evaluations and samples."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters

    final_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name, "final")
    checkpoint_path = os.path.join(final_dir, "model.pth")
    model, checkpoint = load_model_from_checkpoint(checkpoint_path)
    id2label = checkpoint["id2label"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = InstancePredictor(
        model,
        device,
        checkpoint["image_size"],
        tuple(checkpoint["image_mean"]),
        tuple(checkpoint["image_std"]),
        mask_threshold=hp.mask_threshold,
        overlap_mask_area_threshold=hp.overlap_mask_area_threshold,
    )
    log.info(
        "Evaluating %s | %d classes | image_size=%d",
        checkpoint_path,
        len(id2label),
        checkpoint["image_size"],
    )

    ds = picsellia_datasets["test"]
    log.info(
        "Test set: %d images | metrics at score >= %g, Picsellia evaluations at score >= %g, mask threshold %g, batch size %d",
        len(ds.assets),
        hp.metrics_score_threshold,
        hp.score_threshold,
        hp.mask_threshold,
        hp.eval_batch_size,
    )

    # Ground-truth ids: assets are matched to COCO images by file name, and classes
    # to COCO categories by name.
    file_name_to_image_id = {
        img["file_name"]: img["id"] for img in ds.coco_data["images"]
    }
    name2cat_id = {cat["name"]: cat["id"] for cat in ds.coco_data["categories"]}
    cat_id_to_name = {cat_id: name for name, cat_id in name2cat_id.items()}
    class_to_cat_id = {
        index: name2cat_id[name]
        for index, name in id2label.items()
        if name in name2cat_id
    }

    inference = _run_test_inference(
        ds, predictor, class_to_cat_id, file_name_to_image_id, hp
    )
    detections = inference.detections
    log.info(
        "Inference done: %d masks on %d/%d images | model %.1f ms per image, %.1f ms per image with loading and encoding",
        len(detections),
        len({d["image_id"] for d in detections}),
        len(inference.image_ids),
        inference.predict_ms,
        inference.total_ms,
    )

    # 1. COCO metrics on every detection above the low metrics threshold.
    result = evaluate_detections(
        gt_coco_path=ds.coco_file_path,
        detections=detections,
        image_ids=inference.image_ids,
    )
    output_dir = os.path.join(ctx.working_dir, "evaluation")
    files = write_evaluation_files(output_dir, detections, result)

    experiment = ctx.experiment
    for name, value in result["values"].items():
        experiment.log(name=f"test/{name}", data=round(value, 4), type=LogType.VALUE)
    experiment.log(name="test/metrics", data=result["table"], type=LogType.TABLE)
    for name, value in {
        "test/inference_time_ms": inference.predict_ms,
        "test/time_per_image_ms": inference.total_ms,
        "test/images_evaluated": len(inference.image_ids),
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
            image_id_to_asset=inference.assets,
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
        inference.paths,
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

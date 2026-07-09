from __future__ import annotations

import gc
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from picsellia import Experiment
from picsellia.types.enums import LogType
from picsellia_cv_engine.core.models import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)
from PIL import Image
from torch.utils.data import DataLoader

from utils.criterion import Sam3SetCriterion
from utils.data import Sam3SegmentationDataset, collate_fn


def load_sam3_model_and_processor(device: str):
    """Load the SAM-3 model and processor from Hugging Face.

    Mirrors the cache/auth handling used by the SAM-3 pre-annotation pipeline so
    the gated ``facebook/sam3`` weights download cleanly inside the container.
    """
    from dotenv import load_dotenv
    from huggingface_hub import login
    from transformers import Sam3Model, Sam3Processor

    # Reduce CUDA fragmentation so the caching allocator can reuse "reserved but
    # unallocated" blocks instead of OOMing. Read at the first CUDA allocation
    # (the .to(device) below); setdefault lets an explicit env override win.
    # torch>=2.9 renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    hf_cache_dir = os.environ.get("HF_HOME")
    if not hf_cache_dir:
        hf_cache_dir = os.path.join(tempfile.gettempdir(), "huggingface")
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.makedirs(hf_cache_dir, exist_ok=True)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # StreamToLogger used by the pipeline lacks isatty(); huggingface_hub expects it.
    if not hasattr(sys.stdout, "isatty"):
        sys.stdout.isatty = lambda: False  # type: ignore[attr-defined]
    if not hasattr(sys.stderr, "isatty"):
        sys.stderr.isatty = lambda: False  # type: ignore[attr-defined]

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
        except Exception as e:
            print(f"Could not login to Hugging Face: {e}")

    print(f"Loading SAM-3 model on device: {device}")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("SAM-3 model loaded.")
    return model, processor


def configure_trainable_parameters(model, hp) -> list[torch.nn.Parameter]:
    """Freeze the encoders requested by hyperparameters and return trainable params."""
    if hp.freeze_vision_encoder:
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
    if hp.freeze_text_encoder:
        for p in model.text_encoder.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_trainable:,} / {total:,}")
    return trainable


def _forward_sam3(model, batch: dict, hp, device: str):
    """Run SAM-3, skipping gradients through any frozen encoder to save memory."""
    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    if hp.freeze_vision_encoder:
        with torch.no_grad():
            vision_embeds = model.get_vision_features(pixel_values=pixel_values)
        vision_kwargs = {"vision_embeds": vision_embeds}
    else:
        vision_kwargs = {"pixel_values": pixel_values}

    if hp.freeze_text_encoder:
        with torch.no_grad():
            text_embeds = model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
        text_kwargs = {"text_embeds": text_embeds}
    else:
        text_kwargs = {"input_ids": input_ids}

    return model(**vision_kwargs, **text_kwargs, attention_mask=attention_mask)


def _make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda


def _log_line(experiment: Experiment, name: str, value: float) -> None:
    experiment.log(name=name, data=float(value), type=LogType.LINE)


def train_sam3(
    model,
    processor,
    datasets,
    hp,
    experiment: Experiment,
    output_dir: str,
    device: str,
) -> str:
    """Fine-tune SAM-3 and store the resulting weights on the experiment.

    Returns the directory containing the saved fine-tuned model.
    """
    trainable_params = configure_trainable_parameters(model, hp)

    train_ds = Sam3SegmentationDataset(
        images_dir=datasets["train"].images_dir,
        coco_file_path=datasets["train"].coco_file_path,
        processor=processor,
        mask_resolution=hp.mask_resolution,
        include_negatives=hp.include_negative_samples,
    )
    print(f"Training samples ((image, concept) pairs): {len(train_ds)}")
    print(f"Concepts: {train_ds.concept_names}")

    # num_workers=0 by default: on Python 3.14 the multiprocessing start method
    # is "forkserver", and spawning DataLoader workers while CUDA, a HF Rust
    # tokenizer, and the Picsellia client are all live in the parent deadlocks
    # (the pipeline "freezes" on the first batch). Loading in the main process
    # avoids it. Opt back into workers with WORKERS=N once that's not a concern.
    workers = int(os.getenv("WORKERS", "0"))
    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
    )

    criterion = Sam3SetCriterion(hp).to(device)
    optimizer = torch.optim.AdamW(
        trainable_params, lr=hp.learning_rate, weight_decay=hp.weight_decay
    )

    accum = max(1, hp.grad_accumulation_steps)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = max(1, steps_per_epoch * hp.epochs)
    warmup_steps = int(hp.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _make_lr_lambda(warmup_steps, total_steps)
    )

    experiment.log_parameters(hp.to_dict())

    # bf16 autocast roughly halves forward-activation memory, which is what lets
    # SAM-3's 1008x1008 forward fit on a 15 GB GPU. bf16 keeps fp32's exponent
    # range, so no GradScaler is needed; matmul-heavy ops run in bf16 while
    # numerically sensitive ops (BCE, etc.) are auto-promoted to fp32 internally.
    use_amp = device.startswith("cuda")

    # On a T4 a single SAM-3 step is slow (its vision attention runs on SDPA's
    # math kernel), so a 1661-sample epoch can take a long time with no output
    # and look hung. Log per-step progress (flushed) so the run is visibly alive.
    log_every = max(1, int(os.getenv("LOG_EVERY", "10")))
    print(
        f"Starting training: {len(train_loader)} batches/epoch x {hp.epochs} epochs",
        flush=True,
    )

    global_step = 0
    try:
        for epoch in range(hp.epochs):
            model.train()
            epoch_losses: dict[str, float] = {}
            num_batches = 0
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                step_t0 = time.time()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
                ):
                    outputs = _forward_sam3(model, batch, hp, device)
                    losses = criterion(outputs, batch["targets"])
                loss = losses["loss"] / accum
                loss.backward()

                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, hp.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if i % log_every == 0:
                    mem = (
                        torch.cuda.max_memory_allocated() / 1024**3
                        if device.startswith("cuda")
                        else 0.0
                    )
                    print(
                        f"  epoch {epoch + 1}/{hp.epochs} "
                        f"batch {i + 1}/{len(train_loader)} "
                        f"loss={float(losses['loss']):.4f} "
                        f"{time.time() - step_t0:.2f}s/it peak={mem:.1f}GiB",
                        flush=True,
                    )

                for name, value in losses.items():
                    epoch_losses[name] = (
                        epoch_losses.get(name, 0.0) + float(value.detach())
                    )
                num_batches += 1

            avg = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
            print(
                f"Epoch {epoch + 1}/{hp.epochs} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
            )
            for name, value in avg.items():
                _log_line(experiment, f"train/{name}", value)
            _log_line(experiment, "learning_rate", scheduler.get_last_lr()[0])
    finally:
        # Deterministically tear down the DataLoader's worker processes. A CUDA
        # OOM (or any error) mid-iteration otherwise leaves workers orphaned, and
        # they leak the POSIX semaphores / shared memory they allocated
        # ("resource_tracker: leaked semaphore objects"). Dropping the loader and
        # forcing a GC runs the iterator's __del__, which joins the workers and
        # unlinks those resources before the process exits.
        del train_loader
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return _save_and_upload(model, processor, experiment, output_dir)


def _save_and_upload(
    model, processor, experiment: Experiment, output_dir: str
) -> str:
    """Save the fine-tuned model with the HF format and store it on the experiment."""
    save_dir = os.path.join(output_dir, "sam3-finetuned")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving fine-tuned SAM-3 to {save_dir}")
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    zip_path = shutil.make_archive(save_dir, "zip", save_dir)
    print(f"Uploading model archive {zip_path} to the experiment")
    experiment.store(name="model-latest", path=zip_path)
    return save_dir


def _mask_to_polygon(mask: np.ndarray, min_area: float) -> list[list[int]] | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    simplified = cv2.approxPolyDP(contour, 2.0, True)
    if len(simplified) < 3:
        return None
    return [[int(pt[0][0]), int(pt[0][1])] for pt in simplified]


@torch.no_grad()
def run_sam3_inference(
    model, processor, dataset, hp, device: str
) -> list[PicselliaPolygonPrediction]:
    """Run the fine-tuned model over a dataset split, one concept prompt at a time."""
    model.eval()
    concepts = list(dataset.labelmap.keys())
    predictions: list[PicselliaPolygonPrediction] = []

    for asset in dataset.assets:
        img_path = os.path.join(dataset.images_dir, asset.id_with_extension)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        target_size = (image.size[1], image.size[0])  # (height, width)

        polygons: list[PicselliaPolygon] = []
        labels: list[PicselliaLabel] = []
        confidences: list[PicselliaConfidence] = []

        for concept in concepts:
            inputs = processor(images=image, text=concept, return_tensors="pt").to(
                device
            )
            outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=hp.eval_score_threshold,
                mask_threshold=hp.eval_mask_threshold,
                target_sizes=[target_size],
            )[0]

            label = PicselliaLabel(dataset.dataset_version.get_or_create_label(concept))
            for mask, score in zip(results["masks"], results["scores"]):
                polygon_points = _mask_to_polygon(
                    mask.cpu().numpy(), hp.eval_min_polygon_area
                )
                if polygon_points is None:
                    continue
                polygons.append(PicselliaPolygon(points=polygon_points))
                labels.append(label)
                confidences.append(PicselliaConfidence(value=float(score)))

        if polygons:
            predictions.append(
                PicselliaPolygonPrediction(
                    asset=asset,
                    polygons=polygons,
                    labels=labels,
                    confidences=confidences,
                )
            )

    return predictions

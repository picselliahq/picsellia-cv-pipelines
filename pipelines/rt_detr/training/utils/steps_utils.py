from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from picsellia import Experiment
from picsellia.types.enums import LogType
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.core.models import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaRectangle,
    PicselliaRectanglePrediction,
)
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainerCallback,
    TrainingArguments,
)


class CocoHFDataset(Dataset):
    """HF-ready dataset from COCO JSON and image directory."""

    def __init__(
        self, images_dir: str, ann_json: str, processor: Any, keep_crowd: bool = False
    ) -> None:
        self.images_dir = images_dir
        self.coco = COCO(ann_json)
        self.processor = processor
        self.keep_crowd = keep_crowd
        self.img_ids: list[int] = list(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        image = Image.open(os.path.join(self.images_dir, info["file_name"])).convert(
            "RGB"
        )
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        if not self.keep_crowd:
            anns = [a for a in anns if int(a.get("iscrowd", 0)) == 0]
        encoding = self.processor(
            image,
            annotations={
                "image_id": int(img_id),
                "annotations": [
                    {
                        "bbox": a["bbox"],
                        "category_id": int(a["category_id"]),
                        "area": float(a.get("area", a["bbox"][2] * a["bbox"][3])),
                        "iscrowd": int(a.get("iscrowd", 0)),
                    }
                    for a in anns
                ],
            },
            return_tensors="pt",
        )
        return {
            k: (
                v.squeeze(0)
                if isinstance(v, torch.Tensor)
                else (
                    v[0] if k == "labels" and isinstance(v, list) and len(v) == 1 else v
                )
            )
            for k, v in encoding.items()
        }


def hf_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch collator for detection tasks."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def build_label_maps(ds: CocoDataset) -> tuple[dict[int, str], dict[str, int]]:
    """Return id2label and label2id mappings from a CocoDataset labelmap."""
    id2label = dict(enumerate(ds.labelmap.keys()))
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def load_processor_and_model(
    hf_ckpt: str, num_labels: int, id2label: dict[int, str], label2id: dict[str, int]
):
    """Instantiate processor and model for object detection with a custom label space."""
    processor = AutoImageProcessor.from_pretrained(hf_ckpt)
    model = AutoModelForObjectDetection.from_pretrained(
        hf_ckpt,
        ignore_mismatched_sizes=True,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return processor, model


def build_datasets(
    datasets: DatasetCollection[CocoDataset], processor: Any
) -> tuple[CocoHFDataset, CocoHFDataset]:
    """Create train and validation datasets from a DatasetCollection."""
    train_ds = CocoHFDataset(
        images_dir=datasets["train"].images_dir,
        ann_json=datasets["train"].coco_file_path,
        processor=processor,
    )
    val_ds = CocoHFDataset(
        images_dir=datasets["val"].images_dir,
        ann_json=datasets["val"].coco_file_path,
        processor=processor,
    )
    return train_ds, val_ds


def build_training_args(
    out_dir: str,
    batch_size: int,
    epochs: int,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.05,
    workers: int = 4,
) -> TrainingArguments:
    """Return standard TrainingArguments for fine-tuning."""
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=min(batch_size, 8),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=workers,
        remove_unused_columns=False,
        report_to=[],
    )


def save_and_upload_artifacts(
    picsellia_model: Model, experiment: Experiment, processor: Any, model: Any
) -> None:
    """Save weights to a 'final' folder, zip it, and upload to the experiment."""
    out_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name)
    final_dir = os.path.join(out_dir, "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    archive_base = os.path.join(
        out_dir, f"{picsellia_model.name}_final_{int(time.time())}"
    )
    archive_path = shutil.make_archive(archive_base, "zip", final_dir)
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"Archive not created: {archive_path}")
    picsellia_model.save_artifact_to_experiment(
        experiment=experiment,
        artifact_name="best-model",
        artifact_path=archive_path,
    )


def run_inference_on_asset(
    ds: CocoDataset,
    asset,
    processor: Any,
    model: Any,
    device: torch.device,
    id2label: dict[int, str],
    conf_thresh: float = 0.25,
) -> PicselliaRectanglePrediction | None:
    """Return a PicselliaRectanglePrediction for a single asset or None."""
    img_path = os.path.join(ds.images_dir, asset.id_with_extension)
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    det = processor.post_process_object_detection(
        outputs, threshold=conf_thresh, target_sizes=[(h, w)]
    )[0]
    if len(det["boxes"]) == 0:
        return None
    boxes: list[PicselliaRectangle] = []
    labels: list[PicselliaLabel] = []
    confidences: list[PicselliaConfidence] = []
    for box, score, label_id in zip(
        det["boxes"], det["scores"], det["labels"], strict=False
    ):
        x1, y1, x2, y2 = box.tolist()
        boxes.append(PicselliaRectangle(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        name = id2label.get(int(label_id), str(int(label_id)))
        labels.append(PicselliaLabel(ds.dataset_version.get_or_create_label(name)))
        confidences.append(PicselliaConfidence(float(score)))
    return PicselliaRectanglePrediction(
        asset=asset, boxes=boxes, labels=labels, confidences=confidences
    )


class PicselliaLogger(TrainerCallback):
    """Push selected Trainer logs to Picsellia."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def on_log(
        self, args, state, control, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        if not logs:
            return
        for k in ("loss", "learning_rate", "grad_norm", "eval_loss", "eval_runtime"):
            v = logs.get(k)
            if isinstance(v, int | float):
                self.experiment.log(name=k, data=float(v), type=LogType.LINE)

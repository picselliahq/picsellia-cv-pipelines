import os
from typing import Any

import torch
from picsellia.types.enums import InferenceType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.core.models import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaRectangle,
    PicselliaRectanglePrediction,
)
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)


# -------------------------
# Minimal COCO → HF dataset
# -------------------------
class CocoHFDataset(Dataset):
    """Loads images + COCO anns, lets the HF image processor build targets."""

    def __init__(
        self, images_dir: str, ann_json: str, processor, keep_crowd: bool = False
    ):
        self.images_dir = images_dir
        self.coco = COCO(ann_json)
        self.processor = processor
        self.keep_crowd = keep_crowd
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.images_dir, info["file_name"])
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        if not self.keep_crowd:
            anns = [a for a in anns if int(a.get("iscrowd", 0)) == 0]

        # HF expects COCO xywh + category_id
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
        # squeeze batch dim added by processor
        return {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in encoding.items()
        }


def hf_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack pixel_values; keep labels as a list of dicts."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# -----------
# Train step
# -----------
@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    """
    Train RT-DETR (or RT-DETRv2) using Hugging Face Transformers Trainer on COCO data.
    Saves processor+model into results_dir/<model_name>/final and uploads as 'best-model'.
    """
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters

    # 1) Checkpoint HF (fixe, on n'utilise pas pretrained_weights_path)
    HF_CKPT = "PekingU/rtdetr_v2_r50vd"

    # 2) Label maps
    id2label = dict(enumerate(picsellia_datasets["train"].labelmap.keys()))
    label2id = {v: k for k, v in id2label.items()}

    # 3) Processor + modèle
    processor = AutoImageProcessor.from_pretrained(HF_CKPT)
    model = AutoModelForObjectDetection.from_pretrained(
        HF_CKPT,
        ignore_mismatched_sizes=True,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    # 4) Datasets
    train_ds = CocoHFDataset(
        images_dir=picsellia_datasets["train"].images_dir,
        ann_json=picsellia_datasets["train"].coco_file_path,
        processor=processor,
    )
    val_ds = CocoHFDataset(
        images_dir=picsellia_datasets["val"].images_dir,
        ann_json=picsellia_datasets["val"].coco_file_path,
        processor=processor,
    )

    # 5) Training args
    out_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=hp.batch_size,
        per_device_eval_batch_size=min(hp.batch_size, 8),
        num_train_epochs=hp.epochs,
        learning_rate=getattr(hp, "learning_rate", 5e-5),
        weight_decay=getattr(hp, "weight_decay", 0.05),
        warmup_ratio=getattr(hp, "warmup_ratio", 0.05),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=int(os.getenv("WORKERS", "4")),
        remove_unused_columns=False,  # important for detection tasks
        report_to=[],  # plug your logger if needed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=hf_collate,
    )

    trainer.train()

    # 6) Save final + upload to Picsellia
    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    picsellia_model.save_artifact_to_experiment(
        experiment=ctx.experiment,
        artifact_name="best-model",
        artifact_path=final_dir,  # directory with pytorch_model.bin, config.json, preprocessor config, etc.
    )


# --------------
# Evaluate step
# --------------
@step()
def evaluate(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
):
    """
    Inférence RT-DETR sur le split test (ou val), agrégation par asset et
    envoi à evaluate_model_impl via PicselliaRectanglePrediction.
    """
    ctx = Pipeline.get_active_context()

    # 1) Charger modèle + processor sauvegardés pendant le train
    final_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name, "final")
    processor = AutoImageProcessor.from_pretrained(final_dir)
    model = AutoModelForObjectDetection.from_pretrained(final_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 2) Split à évaluer
    ds = picsellia_datasets["test"]
    id2label = dict(enumerate(ds.labelmap.keys()))

    predictions: list[PicselliaRectanglePrediction] = []

    # 3) Boucle assets → un seul PicselliaRectanglePrediction par asset
    for asset in ds.assets:
        img_path = asset.file_path
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        det = processor.post_process_object_detection(
            outputs,
            threshold=0.25,  # à ajuster selon ton use-case
            target_sizes=[(h, w)],
        )[0]

        boxes: list[PicselliaRectangle] = []
        labels: list[PicselliaLabel] = []
        confidences: list[PicselliaConfidence] = []

        # Agréger toutes les détections de l’asset courant
        for box, score, label_id in zip(
            det["boxes"], det["scores"], det["labels"], strict=False
        ):
            x1, y1, x2, y2 = box.tolist()
            # Rectangle attend des entiers [x, y, w, h]
            rect = PicselliaRectangle(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            label_name = id2label.get(int(label_id), str(int(label_id)))
            pics_label = PicselliaLabel(
                ds.dataset_version.get_or_create_label(label_name)
            )

            boxes.append(rect)
            labels.append(pics_label)
            confidences.append(PicselliaConfidence(float(score)))

        # Créer UNE prédiction par asset (même si liste vide → on skip)
        if boxes:
            predictions.append(
                PicselliaRectanglePrediction(
                    asset=asset,
                    boxes=boxes,
                    labels=labels,
                    confidences=confidences,
                )
            )

    # 4) Évaluation Picsellia (mAP, etc.)
    if predictions:
        training_labelmap = (
            ctx.experiment.get_log("labelmap").data
            if ctx.experiment.get_log("labelmap")
            else id2label
        )
        evaluate_model_impl(
            context=ctx,
            picsellia_predictions=predictions,
            inference_type=InferenceType.OBJECT_DETECTION,
            assets=ds.assets,
            output_dir=os.path.join(ctx.working_dir, "evaluation"),
            training_labelmap=training_labelmap,
        )

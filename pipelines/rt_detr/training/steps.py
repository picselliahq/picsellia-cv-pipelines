from __future__ import annotations

import os

import torch
from picsellia.types.enums import InferenceType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.core.models import (
    PicselliaRectanglePrediction,
)
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
)
from utils.steps_utils import (
    PicselliaLogger,
    build_datasets,
    build_label_maps_from_coco,
    build_training_args,
    hf_collate,
    load_processor_and_model,
    run_inference_on_asset,
    save_and_upload_artifacts,
)


@step()
def train(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Train RT-DETR(v2) on COCO datasets and upload zipped weights."""
    ctx = Pipeline.get_active_context()
    hp = ctx.hyperparameters
    id2label, label2id = build_label_maps_from_coco(
        picsellia_datasets["train"].coco_file_path
    )
    processor, model = load_processor_and_model(
        hf_ckpt=hp.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        image_size=hp.image_size,
    )
    train_ds, val_ds = build_datasets(picsellia_datasets, processor)
    out_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name)
    args = build_training_args(
        out_dir=out_dir,
        batch_size=hp.batch_size,
        epochs=hp.epochs,
        learning_rate=getattr(hp, "learning_rate", 5e-5),
        weight_decay=getattr(hp, "weight_decay", 0.05),
        warmup_ratio=getattr(hp, "warmup_ratio", 0.05),
        workers=int(os.getenv("WORKERS", "4")),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=hf_collate,
        callbacks=[PicselliaLogger(ctx.experiment)],
    )
    trainer.train()
    save_and_upload_artifacts(picsellia_model, ctx.experiment, processor, model)


@step()
def evaluate(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Run inference on test split and evaluate in Picsellia."""
    ctx = Pipeline.get_active_context()
    final_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name, "final")
    processor = AutoImageProcessor.from_pretrained(final_dir)
    model = AutoModelForObjectDetection.from_pretrained(final_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ds = picsellia_datasets["test"]
    # Use the label space the model was actually trained with (baked into its config),
    # not one rebuilt from the test set — test-set label ordering is not guaranteed to
    # match the training categories.
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    predictions: list[PicselliaRectanglePrediction] = []
    for asset in ds.assets:
        pred = run_inference_on_asset(
            ds, asset, processor, model, device, id2label, conf_thresh=0.15
        )
        if pred:
            predictions.append(pred)
    if predictions:
        training_labelmap = ctx.experiment.get_log("labelmap").data
        evaluate_model_impl(
            context=ctx,
            picsellia_predictions=predictions,
            inference_type=InferenceType.OBJECT_DETECTION,
            assets=ds.assets,
            output_dir=os.path.join(ctx.working_dir, "evaluation"),
            training_labelmap=training_labelmap,
        )

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from picsellia import Experiment
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoModel
from utils.step_utils import (
    DinoClassifier,
    DinoDataset,
    build_df_from_coco,
    evaluate_model,
    load_trained_model,
    log_picsellia_evaluation,
    prepare_test_data,
    run_inference,
    save_experiment_artifacts,
    train_dinov2,
)


@step()
def train(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Train and evaluate a DINOv2 classifier on train/val datasets."""
    context = Pipeline.get_active_context()
    experiment: Experiment = context.experiment

    model_name: str = getattr(
        context.hyperparameters, "model_name", "facebook/dinov2-small"
    )
    epochs: int = getattr(context.hyperparameters, "epochs", 100)
    lr: float = getattr(context.hyperparameters, "learning_rate", 1e-4)
    batch_size: int = getattr(context.hyperparameters, "batch_size", 256)
    patience: int = getattr(context.hyperparameters, "patience", 5)
    seed: int = getattr(context.hyperparameters, "seed", 42)
    n_blocks: int = getattr(context.hyperparameters, "n_blocks", 3)
    use_bbox_features: bool = getattr(
        context.hyperparameters, "use_bbox_features", True
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = build_df_from_coco(
        coco_data=picsellia_datasets["train"].coco_data,
        use_bbox_features=use_bbox_features,
    )
    val_df = build_df_from_coco(
        coco_data=picsellia_datasets["val"].coco_data,
        use_bbox_features=use_bbox_features,
    )

    le = LabelEncoder()
    all_categories = pd.concat([train_df["category"], val_df["category"]]).unique()
    le.fit(all_categories)
    train_df["label_idx"] = le.transform(train_df["category"])
    val_df["label_idx"] = le.transform(val_df["category"])

    train_loader = DataLoader(
        DinoDataset(
            df=train_df,
            image_dir=picsellia_datasets["train"].images_dir,
            model_key=model_name,
            use_geom=use_bbox_features,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        DinoDataset(
            df=val_df,
            image_dir=picsellia_datasets["val"].images_dir,
            model_key=model_name,
            use_geom=use_bbox_features,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    backbone = AutoModel.from_pretrained(model_name)
    model = DinoClassifier(
        backbone, num_classes=len(le.classes_), use_geom=use_bbox_features
    ).to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for block in model.backbone.encoder.layer[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = train_dinov2(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        patience=patience,
        experiment=experiment,
        output_dir=picsellia_model.results_dir,
    )

    evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        experiment=experiment,
        split_name="val",
    )
    save_experiment_artifacts(picsellia_model=picsellia_model)
    print("✅ Training and validation complete.")


@step()
def evaluate(
    picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]
) -> None:
    """Evaluate a trained DINOv2 model on the test dataset."""
    context = Pipeline.get_active_context()
    experiment: Experiment = context.experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name: str = getattr(
        context.hyperparameters, "hugging_face_model_name", "facebook/dinov2-small"
    )
    batch_size: int = getattr(context.hyperparameters, "batch_size", 64)
    use_bbox_features: bool = getattr(
        context.hyperparameters, "use_bbox_features", True
    )

    test_ctx: CocoDataset = picsellia_datasets["test"]
    train_ctx: CocoDataset = picsellia_datasets["train"]

    df_test, le = prepare_test_data(
        train_ctx=train_ctx, test_ctx=test_ctx, use_bbox_features=use_bbox_features
    )
    test_loader = DataLoader(
        DinoDataset(
            df=df_test,
            image_dir=test_ctx.images_dir,
            model_key=model_name,
            use_geom=use_bbox_features,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    weights_path = os.path.join(picsellia_model.results_dir, "best_dino_classifier.pt")
    model = load_trained_model(
        model_name=model_name,
        num_classes=len(le.classes_),
        use_bbox_features=use_bbox_features,
        weights_path=weights_path,
        device=device,
        experiment=experiment,
    )

    picsellia_predictions = run_inference(
        model=model,
        dataloader=test_loader,
        df_test=df_test,
        test_ctx=test_ctx,
        le=le,
        device=device,
    )
    log_picsellia_evaluation(
        context=context,
        experiment=experiment,
        picsellia_model=picsellia_model,
        test_ctx=test_ctx,
        predictions=picsellia_predictions,
    )
    print("✅ Test evaluation completed successfully.")

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from picsellia import Experiment
from picsellia.types.enums import LogType
from picsellia_cv_engine.core import CocoDataset, Model
from picsellia_cv_engine.core.models import (
    PicselliaClassificationPrediction,
    PicselliaConfidence,
    PicselliaLabel,
)
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel


def _safe_wh_from_bbox(
    ann_bbox: list | tuple, img_w: int, img_h: int
) -> tuple[float, float]:
    """Return bbox width/height or fallback to image size."""
    if isinstance(ann_bbox, list | tuple) and len(ann_bbox) >= 4:
        try:
            w, h = float(ann_bbox[2]), float(ann_bbox[3])
            if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
                return w, h
        except (ValueError, TypeError):
            pass
    return float(img_w), float(img_h)


def build_df_from_coco(coco_data: dict, use_bbox_features: bool = True) -> pd.DataFrame:
    """Convert COCO data to a DataFrame (file, bbox, category)."""
    rows = []
    images_by_id = {img["id"]: img for img in coco_data.get("images", [])}
    cat_name_by_id = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    for ann in coco_data.get("annotations", []):
        img = images_by_id.get(ann.get("image_id"))
        if not img:
            continue
        cat_name = cat_name_by_id.get(ann.get("category_id"), "unknown")
        bw, bh = (
            _safe_wh_from_bbox(
                ann.get("bbox"), img.get("width", 0), img.get("height", 0)
            )
            if use_bbox_features
            else (None, None)
        )
        rows.append(
            {
                "file_name": img["file_name"],
                "bbox_width": bw,
                "bbox_height": bh,
                "image_width": img.get("width", 0),
                "image_height": img.get("height", 0),
                "category": cat_name,
            }
        )

    df = pd.DataFrame(rows)
    if use_bbox_features:
        for col in ["bbox_height", "bbox_width"]:
            df[col] = df[col].fillna(df["image_" + col.split("_")[1]])
            cmin, cmax = df[col].min(), df[col].max()
            df[col] = (df[col] - cmin) / max(cmax - cmin, 1e-12)
    return df


def torch_preprocess(img: torch.Tensor) -> torch.Tensor:
    """Normalize and center-crop an image tensor."""
    if img.dtype == torch.uint8:
        img = img.to(torch.float32) / 255.0
    if img.dim() == 3:
        img = img.unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device)
    crop_size = 224
    _, _, h, w = img.shape
    scale = max(crop_size / min(h, w), 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    img = nn.functional.interpolate(
        img, size=(new_h, new_w), mode="bicubic", align_corners=False
    )
    top, left = (new_h - crop_size) // 2, (new_w - crop_size) // 2
    img = img[:, :, top : top + crop_size, left : left + crop_size]
    img = (img - mean) / std
    return img.squeeze(0)


class DinoDataset(Dataset):
    """Custom dataset with optional bbox features."""

    def __init__(
        self, df: pd.DataFrame, image_dir: str, model_key: str, use_geom: bool
    ):
        self.image_dir = image_dir
        self.model_key = model_key
        self.use_geom = use_geom

        # Vérifie l'existence de chaque image
        valid_rows = []
        for idx, row in df.iterrows():
            image_path = os.path.join(image_dir, row["file_name"])
            if os.path.exists(image_path):
                valid_rows.append(row)
            else:
                print(f"⚠️ Missing image: {image_path}")
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["file_name"])
        image = Image.open(image_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(image).transpose(2, 0, 1))
        pixel_values = torch_preprocess(img_tensor)
        label = int(row["label_idx"])
        bbox_features = None
        if self.use_geom:
            bbox_features = torch.tensor(
                [float(row["bbox_height"]), float(row["bbox_width"])],
                dtype=torch.float32,
            )
        return pixel_values, bbox_features, label


class DinoClassifier(nn.Module):
    """DINOv2 classifier with optional geometric input."""

    def __init__(self, backbone: nn.Module, num_classes: int, use_geom: bool):
        super().__init__()
        self.backbone = backbone
        self.use_geom = use_geom
        extra = 2 if use_geom else 0
        self.fc = nn.Linear(backbone.config.hidden_size + extra, num_classes)

    def forward(
        self, x: torch.Tensor, bbox_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.backbone(x)
        cls_token = out.last_hidden_state[:, 0, :]
        x_cat = (
            torch.cat([cls_token, bbox_features], dim=1)
            if self.use_geom and bbox_features is not None
            else cls_token
        )
        return self.fc(x_cat)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    experiment: Experiment,
    split_name: str = "val",
) -> None:
    """Evaluate a model and log accuracy."""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for pixel_values, bbox_features, y in tqdm(
            dataloader, desc=f"Evaluating {split_name}"
        ):
            pixel_values, y = pixel_values.to(device), y.to(device)
            if bbox_features is not None:
                bbox_features = bbox_features.to(device)
            logits = model(pixel_values, bbox_features)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())

    if not preds:
        print(f"⚠️ No predictions for {split_name}.")
        return

    y_pred, y_true = np.concatenate(preds), np.concatenate(labels)
    acc = float((y_pred == y_true).mean())
    experiment.log(name=f"{split_name}/accuracy", data=acc, type=LogType.VALUE)
    print(f"✅ Eval done on {split_name} — accuracy={acc:.4f}")


def save_experiment_artifacts(picsellia_model: Model, experiment: Experiment) -> None:
    """Upload only model weights as 'model-latest' artifact."""
    weights_path = os.path.join(picsellia_model.results_dir, "best_dino_classifier.pt")
    if os.path.exists(weights_path):
        picsellia_model.save_artifact_to_experiment(
            experiment=experiment,
            artifact_name="model-latest",
            artifact_path=weights_path,
        )
        print(f"📦 Uploaded 'model-latest' → {weights_path}")
    else:
        print("⚠️ No model weights found to upload.")
    print("✅ Training and validation complete.")


def train_dinov2(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    experiment: Experiment,
    output_dir: str,
) -> nn.Module:
    """Train the DINOv2 model with early stopping."""
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for pixel_values, bbox_features, y in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}"
        ):
            pixel_values, y = pixel_values.to(device), y.to(device)
            if bbox_features is not None:
                bbox_features = bbox_features.to(device)
            optimizer.zero_grad()
            loss = criterion(model(pixel_values, bbox_features), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pixel_values.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pixel_values, bbox_features, y in val_loader:
                pixel_values, y = pixel_values.to(device), y.to(device)
                if bbox_features is not None:
                    bbox_features = bbox_features.to(device)
                loss = criterion(model(pixel_values, bbox_features), y)
                val_loss += loss.item() * pixel_values.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} → train={avg_train_loss:.4f} | val={avg_val_loss:.4f}"
        )

        experiment.log(name="train/loss", data=avg_train_loss, type=LogType.LINE)
        experiment.log(name="val/loss", data=avg_val_loss, type=LogType.LINE)

        if avg_val_loss < best_val_loss:
            best_val_loss, no_improve = avg_val_loss, 0
            torch.save(
                model.state_dict(), os.path.join(output_dir, "best_dino_classifier.pt")
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    print(f"✅ Training finished — best val loss={best_val_loss:.4f}")
    return model


def prepare_test_data(
    train_ctx: CocoDataset, test_ctx: CocoDataset, use_bbox_features: bool
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Create train/test DataFrames and label encoder."""
    df_train = build_df_from_coco(train_ctx.coco_data, use_bbox_features)
    df_test = build_df_from_coco(test_ctx.coco_data, use_bbox_features)
    le = LabelEncoder()
    le.fit(pd.concat([df_train["category"], df_test["category"]]).unique())
    df_test["label_idx"] = le.transform(df_test["category"])
    return df_test, le


def load_trained_model(
    model_name: str,
    num_classes: int,
    use_bbox_features: bool,
    weights_path: str,
    device: torch.device,
    experiment: Experiment,
) -> nn.Module:
    """Load pretrained DINOv2 model and weights."""
    backbone = AutoModel.from_pretrained(model_name)
    model = DinoClassifier(backbone, num_classes, use_bbox_features).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location=device), strict=True
        )
        print(f"✅ Loaded weights from {weights_path}")
    else:
        experiment.log(name="warning/no_best_weights_found", data=1, type=LogType.LINE)
        print("⚠️ No weights found — evaluating untrained model.")
    return model


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    df_test: pd.DataFrame,
    test_ctx: CocoDataset,
    le: LabelEncoder,
    device: torch.device,
) -> list:
    """Run inference and collect predictions."""
    model.eval()
    y_true, y_pred, predictions = [], [], []
    with torch.no_grad():
        for pixel_values, bbox_features, labels in tqdm(
            dataloader, desc="Evaluating test"
        ):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            if bbox_features is not None:
                bbox_features = bbox_features.to(device)
            logits = model(pixel_values, bbox_features)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            for i in range(pixel_values.size(0)):
                try:
                    fname = df_test.iloc[len(y_true) - len(labels) + i]["file_name"]
                    asset_id = os.path.splitext(fname)[0]
                    assets = test_ctx.dataset_version.list_assets(ids=[asset_id])
                    if not assets:
                        continue
                    asset = assets[0]
                    label = PicselliaLabel(
                        test_ctx.dataset_version.get_or_create_label(
                            le.classes_[int(preds[i].item())]
                        )
                    )
                    conf = PicselliaConfidence(float(probs[i, preds[i]].item()))
                    predictions.append(
                        PicselliaClassificationPrediction(
                            asset=asset, label=label, confidence=conf
                        )
                    )
                except Exception as e:
                    print(f"⚠️ Skipping asset {fname}: {e}")
    return predictions


def log_picsellia_evaluation(
    context,
    experiment: Experiment,
    picsellia_model: Model,
    test_ctx: CocoDataset,
    predictions: list,
) -> None:
    """Send predictions to Picsellia for evaluation."""
    if not predictions:
        print("⚠️ No predictions to log.")
        return
    evaluate_model_impl(
        context=context,
        picsellia_predictions=predictions,
        inference_type=picsellia_model.model_version.type,
        assets=test_ctx.assets,
        output_dir=os.path.join(context.working_dir, "evaluation"),
        training_labelmap=experiment.get_log("labelmap").data,
    )
    print(f"📊 Logged {len(predictions)} predictions to Picsellia.")

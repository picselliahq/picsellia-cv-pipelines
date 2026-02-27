from __future__ import annotations

import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
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
from PIL import Image, UnidentifiedImageError


class YOLOV8StyleOutput:
    """Converts YOLOX output format to a normalized bounding box format.

    YOLOX outputs [x1, y1, x2, y2, obj_conf, cls_conf, cls_id] in pixel coords.
    This wrapper normalizes coordinates to [0, 1] and computes final confidence.
    """

    class Boxes:
        def __init__(self, boxes, conf, cls):
            self.xyxyn = boxes
            self.conf = conf
            self.cls = cls

    def __init__(self, yolox_output: torch.Tensor, img_info: dict):
        self.img_width = img_info["width"]
        self.img_height = img_info["height"]
        self.ratio = img_info["ratio"]

        boxes = yolox_output[:, 0:4].clone()
        boxes /= self.ratio

        normalized_boxes = torch.zeros_like(boxes)
        normalized_boxes[:, 0] = boxes[:, 0] / self.img_width
        normalized_boxes[:, 1] = boxes[:, 1] / self.img_height
        normalized_boxes[:, 2] = boxes[:, 2] / self.img_width
        normalized_boxes[:, 3] = boxes[:, 3] / self.img_height

        conf = yolox_output[:, 4] * yolox_output[:, 5]
        cls = yolox_output[:, 6]

        self.boxes = self.Boxes(normalized_boxes, conf, cls)

    @property
    def probs(self):
        return self.boxes.conf


def build_label_maps(ds: CocoDataset) -> tuple[dict[int, str], dict[str, int]]:
    """Return id2label and label2id mappings from a CocoDataset labelmap."""
    id2label = dict(enumerate(ds.labelmap.keys()))
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def open_asset_as_array(asset) -> np.ndarray:
    """Download and open a Picsellia asset as a numpy RGB array."""
    image = Image.open(requests.get(asset.reset_url(), stream=True).raw)
    if hasattr(image, "_getexif") and image._getexif():
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def run_inference_on_asset(
    ds: CocoDataset,
    asset,
    predictor,
    id2label: dict[int, str],
    conf_thresh: float = 0.1,
) -> PicselliaRectanglePrediction | None:
    """Run YOLOX inference on a single asset and return rectangle predictions.

    Args:
        ds: CocoDataset containing the asset.
        asset: Picsellia asset to run inference on.
        predictor: YOLOX Predictor instance.
        id2label: Mapping from class ID to class name.
        conf_thresh: Confidence threshold for detections.

    Returns:
        PicselliaRectanglePrediction or None if no valid detections.
    """
    try:
        image = open_asset_as_array(asset)
    except (UnidentifiedImageError, Exception):
        logging.warning(f"Can't evaluate {asset.filename}, error opening the image")
        return None

    prediction, img_info = predictor.inference(image)

    if prediction[0] is None:
        return None

    yolox_output = YOLOV8StyleOutput(yolox_output=prediction[0], img_info=img_info)

    boxes_list = []
    labels_list = []
    confidences_list = []

    img_width = img_info["width"]
    img_height = img_info["height"]

    num_detections = min(100, len(yolox_output.boxes.conf))
    for i in range(num_detections):
        confidence = float(yolox_output.boxes.conf[i])
        if confidence < conf_thresh:
            continue

        cls_id = int(yolox_output.boxes.cls[i])
        if cls_id not in id2label:
            continue

        # Convert normalized xyxy to x, y, w, h in pixels
        x1 = float(yolox_output.boxes.xyxyn[i, 0]) * img_width
        y1 = float(yolox_output.boxes.xyxyn[i, 1]) * img_height
        x2 = float(yolox_output.boxes.xyxyn[i, 2]) * img_width
        y2 = float(yolox_output.boxes.xyxyn[i, 3]) * img_height

        x1 = max(0.0, min(x1, float(img_width - 1)))
        y1 = max(0.0, min(y1, float(img_height - 1)))
        x2 = max(0.0, min(x2, float(img_width)))
        y2 = max(0.0, min(y2, float(img_height)))

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        rect = PicselliaRectangle(
            int(round(x1)), int(round(y1)), int(round(w)), int(round(h))
        )
        name = id2label[cls_id]
        label = PicselliaLabel(ds.dataset_version.get_or_create_label(name))
        conf = PicselliaConfidence(confidence)

        boxes_list.append(rect)
        labels_list.append(label)
        confidences_list.append(conf)

    if not boxes_list:
        return None

    return PicselliaRectanglePrediction(
        asset=asset,
        boxes=boxes_list,
        labels=labels_list,
        confidences=confidences_list,
    )


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    image_size: int,
    enable_dynamic_axis: bool = False,
    device: torch.device | None = None,
) -> str:
    """Export YOLOX model to ONNX format.

    Args:
        model: Trained YOLOX model.
        output_path: Path to save the ONNX model.
        image_size: Input image size for the model.
        enable_dynamic_axis: Whether to enable dynamic batch size.
        device: Device the model is on.

    Returns:
        Path to the saved ONNX model.
    """
    from YOLOX.yolox.models.network_blocks import SiLU
    from YOLOX.yolox.utils import replace_module

    # Work on a copy to avoid mutating the original model
    export_model = copy.deepcopy(model)
    export_model = replace_module(export_model, torch.nn.SiLU, SiLU)
    export_model.head.decode_in_inference = False
    export_model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    if device is not None and device.type == "cuda":
        dummy_input = dummy_input.to(device)

    input_names = ["input"]
    output_names = ["output_yolox"]
    dynamic_axes = None

    if enable_dynamic_axis:
        dynamic_axes = {"input": {0: "batch_size"}, "output_yolox": {0: "batch_size"}}

    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    return output_path


def save_and_upload_artifacts(
    picsellia_model: Model,
    experiment: Experiment,
    exp,
    args,
    trainer,
    image_size: int,
    enable_dynamic_axis: bool = False,
) -> None:
    """Save YOLOX checkpoint and ONNX model, then upload to Picsellia.

    Args:
        picsellia_model: Picsellia model object.
        experiment: Picsellia experiment object.
        exp: YOLOX experiment config.
        args: Training arguments.
        trainer: YOLOX trainer with best_epoch info.
        image_size: Image size used for training.
        enable_dynamic_axis: Whether ONNX export uses dynamic axes.
    """
    out_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name)
    final_dir = os.path.join(out_dir, "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)

    # Load best checkpoint (try best → last_epoch → latest)
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    ckpt_file = None
    for candidate in ("best_ckpt.pth", "last_epoch_ckpt.pth", "latest_ckpt.pth"):
        path = os.path.join(file_name, candidate)
        if os.path.isfile(path):
            ckpt_file = path
            break

    if ckpt_file is None:
        raise FileNotFoundError(
            f"No checkpoint found in '{file_name}'. "
            "Expected one of: best_ckpt.pth, last_epoch_ckpt.pth, latest_ckpt.pth"
        )

    print(f"  - Loading checkpoint: {ckpt_file}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)

    model = exp.get_model()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Export to ONNX
    print("  - Exporting to ONNX...")
    onnx_path = os.path.join(final_dir, "best.onnx")
    export_to_onnx(model, onnx_path, image_size, enable_dynamic_axis, device)
    print(f"  - ONNX model saved to: {onnx_path}")

    # Upload ONNX model
    picsellia_model.save_artifact_to_experiment(
        artifact_name="model-latest",
        artifact_path=onnx_path,
    )

    # Upload best checkpoint
    best_checkpoint_path = os.path.join(file_name, "best_ckpt.pth")
    if os.path.isfile(best_checkpoint_path):
        picsellia_model.save_artifact_to_experiment(
            artifact_name="best_ckpt.pth",
            artifact_path=best_checkpoint_path,
        )

    # Upload latest checkpoint
    latest_checkpoint_path = os.path.join(file_name, "last_epoch_ckpt.pth")
    if os.path.isfile(latest_checkpoint_path):
        picsellia_model.save_artifact_to_experiment(
            artifact_name="last-epoch-ckpt",
            artifact_path=latest_checkpoint_path,
        )


class PicselliaLogger:
    """Logger for training metrics to Picsellia."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for name, value in metrics.items():
            if isinstance(value, int | float) and math.isfinite(value):
                self.experiment.log(name=name, data=float(value), type=LogType.LINE)

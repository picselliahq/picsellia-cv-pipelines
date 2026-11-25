from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from picsellia import Experiment
from picsellia.types.enums import LogType
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from utils.evaluation import encode_mask_rle, evaluate_detections
from utils.log import fmt_duration, get_logger

log = get_logger()


class CocoMaskDataset(Dataset):
    """PyTorch dataset for COCO instance segmentation with RLE masks.

    MASK dataset versions are exported with RLE ``segmentation`` dicts. In the
    COCO convention RLE annotations may be flagged ``iscrowd=1``, but here every
    RLE is a regular instance, so crowd filtering is disabled by default and the
    ``iscrowd`` target is always 0.
    """

    def __init__(
        self,
        images_dir: str,
        ann_json: str,
        label2id: dict[str, int],
        transforms: Any = None,
        keep_crowd: bool = True,
    ) -> None:
        self.images_dir = images_dir
        self.coco = COCO(ann_json)
        self.transforms = transforms
        self.keep_crowd = keep_crowd
        # self.img_ids: list[int] = list(self.coco.imgs.keys())
        self.img_ids = []
        for id, img in self.coco.imgs.items():
            if os.path.isfile(os.path.join(self.images_dir, img["file_name"])):
                self.img_ids.append(id)

        # Build mapping from COCO category_id to our sequential label (1-indexed for Mask R-CNN)
        # label2id maps label_name -> 0-indexed id, we need +1 for Mask R-CNN (0 = background)
        coco_categories = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}
        self.coco_cat_to_label = {}
        for coco_cat_id, cat_name in coco_categories.items():
            if cat_name in label2id:
                self.coco_cat_to_label[coco_cat_id] = (
                    label2id[cat_name] + 1
                )  # +1 for background

    @staticmethod
    def _decode_mask(seg: Any, info: dict[str, Any]) -> np.ndarray | None:
        """Decode a COCO ``segmentation`` (RLE dict, or polygons as a fallback)."""
        if not seg:
            return None
        if isinstance(seg, dict):
            return mask_utils.decode(seg)
        rles = mask_utils.frPyObjects(seg, info["height"], info["width"])
        return mask_utils.decode(mask_utils.merge(rles))

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> list[float] | None:
        """Return the tight [x, y, w, h] box around a binary mask, or None if empty."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.images_dir, info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        if not self.keep_crowd:
            anns = [a for a in anns if int(a.get("iscrowd", 0)) == 0]

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            # Remap COCO category_id to our sequential label
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in self.coco_cat_to_label:
                continue  # Skip unknown categories

            mask = self._decode_mask(ann.get("segmentation"), info)
            if mask is None:
                continue  # A MASK dataset needs a mask for every instance

            bbox = ann.get("bbox")
            if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                # Derive the box from the mask when the export did not provide one
                bbox = self._bbox_from_mask(mask)
            if bbox is None:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(self.coco_cat_to_label[coco_cat_id])
            areas.append(float(mask.sum()))
            iscrowd.append(0)
            masks.append(mask)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, info["height"], info["width"]), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        image_tensor = F.to_tensor(image)

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list, list]:
    """Custom collate function for detection tasks."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def build_label_maps(ds: CocoDataset) -> tuple[dict[int, str], dict[str, int]]:
    """Return id2label and label2id mappings from a CocoDataset labelmap."""
    id2label = dict(enumerate(ds.labelmap.keys()))
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def get_maskrcnn_model(
    num_classes: int,
    backbone: str = "resnet50",
    pretrained: bool = True,
    image_size: int = 800,
) -> torch.nn.Module:
    """Create a Mask R-CNN model with custom number of classes.

    ``pretrained=True`` loads the torchvision COCO weights of the whole detector
    (backbone, RPN, heads); the box and mask predictors are then replaced for
    ``num_classes``.

    Args:
        num_classes: Number of classes (including background).
        backbone: Backbone architecture ("resnet50" or "resnet50_v2").
        pretrained: Whether to start from the torchvision COCO weights.
        image_size: Shorter side images are resized to (torchvision ``min_size``).
            ``max_size`` keeps torchvision's default 800:1333 ratio.

    Returns:
        Mask R-CNN model ready for training.
    """
    size_kwargs = {
        "min_size": int(image_size),
        "max_size": int(round(image_size * 1333 / 800)),
    }
    if not pretrained:
        # Without COCO weights torchvision still downloads the ImageNet backbone
        # (~100 MB); here the weights come from a checkpoint, so skip it.
        size_kwargs["weights_backbone"] = None
    if backbone == "resnet50_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn_v2(weights=weights, **size_kwargs)
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(weights=weights, **size_kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def build_datasets(
    datasets: DatasetCollection[CocoDataset],
    label2id: dict[str, int],
) -> tuple[CocoMaskDataset, CocoMaskDataset]:
    """Create train and validation datasets from a DatasetCollection.

    Args:
        datasets: DatasetCollection containing train and val datasets.
        label2id: Mapping from label name to 0-indexed class ID.
    """
    train_ds = CocoMaskDataset(
        images_dir=datasets["train"].images_dir,
        ann_json=datasets["train"].coco_file_path,
        label2id=label2id,
    )
    val_ds = CocoMaskDataset(
        images_dir=datasets["val"].images_dir,
        ann_json=datasets["val"].coco_file_path,
        label2id=label2id,
    )
    return train_ds, val_ds


class MaskRCNNWrapper(torch.nn.Module):
    """Wrapper for Mask R-CNN to make it compatible with TorchScript tracing.

    This wrapper returns raw 28x28 masks from the ROI heads instead of pasted
    masks. This avoids TorchScript tracing issues with torchvision's dynamic
    mask pasting operations.

    Clients must apply a sigmoid to the raw masks, resize each one to its box and
    paste it onto the image (see model_metadata.json).
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        # Access internal components for mask extraction
        self.roi_heads = model.roi_heads

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single image returning raw masks.

        Args:
            image: Input tensor of shape [C, H, W]

        Returns:
            Tuple of (boxes, labels, scores, masks)
            - boxes: [N, 4] in image coordinates
            - labels: [N] class labels
            - scores: [N] confidence scores
            - masks: [N, 1, 28, 28] raw mask logits for the predicted class
              Sigmoid, resize to the box and paste to get a full-size mask.
        """
        # Use the model's standard forward to get boxes, labels, scores
        # This properly handles all the NMS and filtering
        outputs = self.model([image])[0]

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        # Now get raw masks for the detected boxes
        # We need to run the mask head manually to get raw logits
        original_h, original_w = image.shape[-2:]

        # Get features and transformed image info
        images, _ = self.model.transform([image], None)
        features = self.model.backbone(images.tensors)

        # Scale boxes back to feature map coordinates
        scale_h = original_h / images.image_sizes[0][0]
        scale_w = original_w / images.image_sizes[0][1]
        boxes_for_mask = boxes.clone()
        boxes_for_mask[:, 0::2] = boxes_for_mask[:, 0::2] / scale_w
        boxes_for_mask[:, 1::2] = boxes_for_mask[:, 1::2] / scale_h

        # Get mask features and predict
        mask_features = self.roi_heads.mask_roi_pool(
            features, [boxes_for_mask], images.image_sizes
        )
        mask_features = self.roi_heads.mask_head(mask_features)
        mask_logits = self.roi_heads.mask_predictor(mask_features)

        # Select mask for predicted class - shape [N, 1, 28, 28]
        # mask_logits is [N, num_classes, 28, 28], we want [N, 1, 28, 28]
        num_masks = mask_logits.shape[0]
        indices = torch.arange(num_masks, device=mask_logits.device)
        masks = mask_logits[indices, labels].unsqueeze(1)  # [N, 1, 28, 28]

        return boxes, labels, scores, masks


def _patch_paste_masks_for_device(device: torch.device):
    """Patch torchvision's paste_masks functions to use the correct device.

    The original functions create CPU tensors (torch.zeros, torch.ones) without
    specifying device, causing device mismatch errors when running on GPU.
    """
    from torchvision.models.detection import roi_heads

    def _onnx_paste_mask_in_image_patched(mask, box, im_h, im_w):
        # Ensure im_h and im_w are on the same device as box
        if isinstance(im_h, torch.Tensor):
            im_h = im_h.to(box.device)
        if isinstance(im_w, torch.Tensor):
            im_w = im_w.to(box.device)

        one = torch.ones(1, dtype=torch.int64, device=box.device)
        zero = torch.zeros(1, dtype=torch.int64, device=box.device)

        w = box[2] - box[0] + one
        h = box[3] - box[1] + one
        w = torch.max(torch.cat((w, one)))
        h = torch.max(torch.cat((h, one)))

        mask = mask.expand((1, 1, -1, -1))
        mask = torch.nn.functional.interpolate(
            mask, size=(int(h), int(w)), mode="bilinear", align_corners=False
        )
        mask = mask[0][0]

        x_0 = torch.max(torch.cat((box[0].unsqueeze(0), zero)))
        x_1 = torch.min(torch.cat((box[2].unsqueeze(0) + one, im_w.unsqueeze(0))))
        y_0 = torch.max(torch.cat((box[1].unsqueeze(0), zero)))
        y_1 = torch.min(torch.cat((box[3].unsqueeze(0) + one, im_h.unsqueeze(0))))

        unpaded_im_mask = mask[
            (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
        ]

        zeros_y0 = torch.zeros(y_0, im_w, device=box.device)
        zeros_y1 = torch.zeros(im_h - y_1, im_w, device=box.device)
        concat_0 = torch.cat(
            (
                torch.zeros(y_1 - y_0, x_0, device=box.device),
                unpaded_im_mask,
                torch.zeros(y_1 - y_0, im_w - x_1, device=box.device),
            ),
            1,
        )
        im_mask = torch.cat((zeros_y0, concat_0, zeros_y1), 0)
        return im_mask

    def _onnx_paste_masks_in_image_loop_patched(masks, boxes, im_h, im_w):
        res_append = torch.zeros(0, im_h, im_w, device=boxes.device)
        for i in range(masks.size(0)):
            mask_res = _onnx_paste_mask_in_image_patched(
                masks[i][0], boxes[i], im_h, im_w
            )
            mask_res = mask_res.unsqueeze(0)
            res_append = torch.cat((res_append, mask_res))
        return res_append

    # Apply patches
    roi_heads._onnx_paste_mask_in_image = _onnx_paste_mask_in_image_patched
    roi_heads._onnx_paste_masks_in_image_loop = _onnx_paste_masks_in_image_loop_patched


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: str,
    device: torch.device,
    image_size: int = 800,
) -> str:
    """Export model to TorchScript format.

    Mask R-CNN has dynamic components (anchor generation, RPN) that make tracing
    difficult. This function uses tracing with check_trace=False to avoid
    strict validation errors; ``validate_torchscript`` then checks the result.

    Args:
        model: Trained Mask R-CNN model.
        output_path: Path to save the TorchScript model.
        device: Device the model is on.
        image_size: Image size used for tracing (should match training).

    Returns:
        Path to the saved TorchScript model.
    """
    model.eval()

    # Patch torchvision's paste_masks functions to use correct device
    # This fixes "Expected all tensors to be on the same device" errors
    _patch_paste_masks_for_device(device)

    wrapped_model = MaskRCNNWrapper(model)
    wrapped_model.eval()
    wrapped_model.to(device)

    example_input = torch.rand(3, image_size, image_size).to(device)

    log.info(
        "Tracing the model with TorchScript (image_size=%d, %s)", image_size, device
    )
    with torch.no_grad(), warnings.catch_warnings():
        # Tracing Mask R-CNN emits dozens of TracerWarning/UserWarning lines
        warnings.simplefilter("ignore")
        # check_trace=False: dynamic anchor generation gives different values per call
        traced_model = torch.jit.trace(
            wrapped_model, example_input, strict=False, check_trace=False
        )

    traced_model.save(output_path)
    return output_path


def validate_torchscript(
    torchscript_path: str,
    model: torch.nn.Module,
    sample_image: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Compare the exported TorchScript with the eager model on one real image.

    Tracing bakes in the shapes seen at export time, so a model traced on a
    square noise image can behave differently on a real, non-square image. This
    never raises: it returns a report with ``status`` in ``ok`` / ``mismatch`` /
    ``inconclusive`` (no detection on the sample).
    """
    eager = MaskRCNNWrapper(model).eval().to(device)
    traced = torch.jit.load(torchscript_path, map_location=device).eval()
    image = sample_image.to(device)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e_boxes, e_labels, e_scores, _ = eager(image)
        t_boxes, t_labels, t_scores, _ = traced(image)

    report: dict[str, Any] = {
        "detections_eager": int(len(e_boxes)),
        "detections_torchscript": int(len(t_boxes)),
        "image_shape": list(image.shape),
    }
    if len(e_boxes) == 0 and len(t_boxes) == 0:
        report["status"] = "inconclusive"
        return report
    if len(e_boxes) != len(t_boxes):
        report["status"] = "mismatch"
        return report

    box_diff = float((e_boxes - t_boxes).abs().max())
    score_diff = float((e_scores - t_scores).abs().max())
    labels_equal = bool((e_labels == t_labels).all())
    report.update(max_box_diff_px=box_diff, max_score_diff=score_diff)
    report["status"] = (
        "ok" if labels_equal and box_diff <= 1.0 and score_diff <= 1e-3 else "mismatch"
    )
    return report


def build_model_metadata(
    id2label: dict[int, str],
    image_size: int,
    backbone: str,
    score_threshold: float,
    mask_threshold: float,
) -> dict[str, Any]:
    """Describe the exported model so that it can be used without this repository."""
    labels = [id2label[i] for i in sorted(id2label)]
    return {
        "framework": "pytorch",
        "model_type": "maskrcnn",
        "task": "instance segmentation (MASK)",
        "backbone": backbone,
        "labels": labels,
        # 0-based here; the model predicts index + 1 because 0 is the background
        "id2label": {str(i): id2label[i] for i in sorted(id2label)},
        "label_offset": 1,
        "num_classes": len(labels),
        "num_classes_with_background": len(labels) + 1,
        "image_size": image_size,
        "min_size": int(image_size),
        "max_size": int(round(image_size * 1333 / 800)),
        "input": (
            "RGB float32 tensor [3, H, W] with values in [0, 1]; resizing and "
            "normalization are done inside the model"
        ),
        "default_thresholds": {
            "score": score_threshold,
            "mask": mask_threshold,
        },
        "torchscript_output": {
            "format": "tuple (boxes, labels, scores, masks)",
            "boxes": "[N, 4] xyxy in pixels of the input image",
            "labels": "[N] int64, 1-based (0 is the background)",
            "scores": "[N] float",
            "masks": (
                "[N, 1, 28, 28] raw logits of the predicted class: apply a "
                "sigmoid, resize to the box, paste onto the image, then "
                "threshold at the mask threshold"
            ),
        },
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
    }


def save_and_upload_artifacts(
    picsellia_model: Model,
    experiment: Experiment,
    model: torch.nn.Module,
    id2label: dict[int, str],
    image_size: int = 800,
    backbone: str = "resnet50",
    sample_images: list[torch.Tensor] | None = None,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> None:
    """Save the model and upload it to the experiment, safest artifact first.

    Order matters: the PyTorch checkpoint and the metadata are uploaded *before*
    the fragile TorchScript export, so a tracing failure cannot lose the training.
    The export is then validated against the eager model on ``sample_images``
    (the first image that gets a detection decides).
    """
    out_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name)
    final_dir = os.path.join(out_dir, "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(final_dir, "model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(id2label) + 1,
            "id2label": id2label,
            "image_size": image_size,
            "backbone": backbone,
        },
        model_path,
    )
    picsellia_model.save_artifact_to_experiment(
        experiment=experiment,
        artifact_name="checkpoint-latest",
        artifact_path=model_path,
    )
    log.info("Uploaded artifact checkpoint-latest (%s)", model_path)

    metadata_path = os.path.join(final_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            build_model_metadata(
                id2label, image_size, backbone, score_threshold, mask_threshold
            ),
            f,
            indent=2,
        )
    experiment.store("model-metadata", metadata_path)
    log.info("Uploaded artifact model-metadata (%s)", metadata_path)

    device = next(model.parameters()).device
    torchscript_path = os.path.join(final_dir, "model.torchscript")
    export_to_torchscript(model, torchscript_path, device, image_size)
    picsellia_model.save_artifact_to_experiment(
        experiment=experiment,
        artifact_name="model-latest",
        artifact_path=torchscript_path,
    )
    log.info("Uploaded artifact model-latest (%s)", torchscript_path)

    if not sample_images:
        log.warning("No sample image: the TorchScript export is not validated")
        return
    report: dict[str, Any] = {}
    for sample_image in sample_images:
        report = validate_torchscript(torchscript_path, model, sample_image, device)
        if report["status"] != "inconclusive":
            break
    if report["status"] == "ok":
        log.info("TorchScript check OK: %s", report)
        experiment.log("export/torchscript_matches_eager", 1.0, LogType.VALUE)
    elif report["status"] == "mismatch":
        log.warning(
            "TorchScript output DIFFERS from the PyTorch model on a real image: %s",
            report,
        )
        experiment.log("export/torchscript_matches_eager", 0.0, LogType.VALUE)
    else:
        log.warning(
            "TorchScript check inconclusive (no detection on %d image(s)): %s",
            len(sample_images),
            report,
        )


def detections_from_outputs(
    output: dict[str, torch.Tensor],
    image_id: int,
    label_to_cat_id: dict[int, int],
    conf_thresh: float,
    mask_thresh: float,
) -> list[dict[str, Any]]:
    """Turn one torchvision output dict into COCO-style RLE detections.

    Filtering and mask thresholding happen on the device, so only the kept
    detections are moved to the CPU (as uint8 masks).

    Args:
        output: Model output for one image (boxes, labels, scores, masks).
        image_id: COCO id of the image in the ground-truth file.
        label_to_cat_id: Model label (1-based) -> COCO category id.
        conf_thresh: Minimum score to keep a detection.
        mask_thresh: Threshold for the binary mask.
    """
    keep = output["scores"] >= conf_thresh
    if not bool(keep.any()):
        return []
    scores = output["scores"][keep].cpu().numpy()
    labels = output["labels"][keep].cpu().numpy()
    boxes = output["boxes"][keep].cpu().numpy()
    masks = (output["masks"][keep][:, 0] > mask_thresh).to(torch.uint8).cpu().numpy()

    detections: list[dict[str, Any]] = []
    for score, label, box, mask in zip(scores, labels, boxes, masks, strict=True):
        cat_id = label_to_cat_id.get(int(label))
        if cat_id is None or not mask.any():
            continue  # background / unknown label / empty mask
        x1, y1, x2, y2 = (float(v) for v in box)
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "score": float(score),
                "segmentation": encode_mask_rle(mask),
                "box": [x1, y1, x2 - x1, y2 - y1],
            }
        )
    return detections


def run_inference_on_asset(
    ds: CocoDataset,
    asset,
    model: torch.nn.Module,
    device: torch.device,
    label_to_cat_id: dict[int, int],
    image_id: int,
    conf_thresh: float = 0.05,
    mask_thresh: float = 0.5,
) -> tuple[list[dict[str, Any]], float]:
    """Run inference on a single asset.

    Returns:
        ``(detections, forward_seconds)``: the RLE detections (see
        ``detections_from_outputs``) and the time of the model forward alone,
        excluding image loading and mask encoding.
    """
    img_path = os.path.join(ds.images_dir, asset.id_with_extension)
    image = Image.open(img_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model([image_tensor])[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start

    detections = detections_from_outputs(
        output, image_id, label_to_cat_id, conf_thresh, mask_thresh
    )
    return detections, forward_seconds


@torch.no_grad()
def compute_val_map(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    val_ds: CocoMaskDataset,
    gt_coco_path: str,
    device: torch.device,
    conf_thresh: float = 0.05,
    mask_thresh: float = 0.5,
) -> dict[str, float]:
    """COCO mask/box metrics on the validation split (model left in eval mode)."""
    model.eval()
    label_to_cat_id = {label: cat for cat, label in val_ds.coco_cat_to_label.items()}
    detections: list[dict[str, Any]] = []
    for images, targets in data_loader:
        outputs = model([img.to(device) for img in images])
        for output, target in zip(outputs, targets, strict=True):
            detections.extend(
                detections_from_outputs(
                    output,
                    int(target["image_id"]),
                    label_to_cat_id,
                    conf_thresh,
                    mask_thresh,
                )
            )
    return evaluate_detections(gt_coco_path, detections, val_ds.img_ids)["values"]


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


def gpu_memory_gb() -> float | None:
    """Peak GPU memory allocated since the last reset, in GB (None without CUDA)."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1e9


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    log_every: int = 0,
) -> dict[str, float]:
    """Train the model for one epoch, logging progress with an ETA.

    Args:
        log_every: Log a progress line every N iterations (0 = about 10 per epoch).

    Returns:
        Average of the total loss (``loss``) and of each loss component.

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

    totals: dict[str, float] = {}
    start = time.perf_counter()
    for iteration, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = float(losses.item())
        if not math.isfinite(loss_value):
            components = {k: float(v.detach()) for k, v in loss_dict.items()}
            raise FloatingPointError(
                f"Loss is {loss_value} at epoch {epoch + 1}, iteration {iteration} "
                f"({components}). Try a lower learning_rate."
            )

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        totals["loss"] = totals.get("loss", 0.0) + loss_value
        for name, value in loss_dict.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())

        if iteration % log_every == 0 or iteration == n_batches:
            elapsed = time.perf_counter() - start
            rate = iteration / elapsed
            memory = gpu_memory_gb()
            log.info(
                "epoch %d/%d | iter %d/%d | loss %.4f | lr %.2e | %.2f it/s | ETA %s%s",
                epoch + 1,
                epochs,
                iteration,
                n_batches,
                totals["loss"] / iteration,
                optimizer.param_groups[0]["lr"],
                rate,
                fmt_duration((n_batches - iteration) / rate),
                f" | gpu {memory:.1f} GB" if memory is not None else "",
            )

    return {name: total / n_batches for name, total in totals.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute the validation losses (the model must be in train mode to return them).

    Returns:
        Average of the total loss (``loss``) and of each loss component, or an
        empty dict when the validation loader is empty.
    """
    model.train()
    n_batches = len(data_loader)
    if n_batches == 0:
        return {}

    totals: dict[str, float] = {}
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        totals["loss"] = totals.get("loss", 0.0) + float(
            sum(loss for loss in loss_dict.values())
        )
        for name, value in loss_dict.items():
            totals[name] = totals.get(name, 0.0) + float(value)

    return {name: total / n_batches for name, total in totals.items()}

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from picsellia import Experiment
from picsellia.types.enums import LogType
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.core.models import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)
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


class CocoMaskDataset(Dataset):
    """PyTorch dataset for COCO instance segmentation with masks."""

    def __init__(
        self,
        images_dir: str,
        ann_json: str,
        label2id: dict[str, int],
        transforms: Any = None,
        keep_crowd: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.coco = COCO(ann_json)
        self.transforms = transforms
        self.keep_crowd = keep_crowd
        self.img_ids: list[int] = list(self.coco.imgs.keys())

        # Build mapping from COCO category_id to our sequential label (1-indexed for Mask R-CNN)
        # label2id maps label_name -> 0-indexed id, we need +1 for Mask R-CNN (0 = background)
        coco_categories = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}
        self.coco_cat_to_label = {}
        for coco_cat_id, cat_name in coco_categories.items():
            if cat_name in label2id:
                self.coco_cat_to_label[coco_cat_id] = (
                    label2id[cat_name] + 1
                )  # +1 for background

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
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            # Remap COCO category_id to our sequential label
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in self.coco_cat_to_label:
                continue  # Skip unknown categories

            boxes.append([x, y, x + w, y + h])
            labels.append(self.coco_cat_to_label[coco_cat_id])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

            if "segmentation" in ann:
                seg = ann["segmentation"]
                if isinstance(seg, dict):
                    mask = mask_utils.decode(seg)
                else:
                    rle = mask_utils.frPyObjects(seg, info["height"], info["width"])
                    mask = mask_utils.decode(mask_utils.merge(rle))
                masks.append(mask)
            else:
                mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
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
) -> torch.nn.Module:
    """Create a Mask R-CNN model with custom number of classes.

    Args:
        num_classes: Number of classes (including background).
        backbone: Backbone architecture ("resnet50" or "resnet50_v2").
        pretrained: Whether to use pretrained weights.

    Returns:
        Mask R-CNN model ready for training.
    """
    if backbone == "resnet50_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(weights=weights)

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


def paste_masks_in_image_gpu(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    img_h: int,
    img_w: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Paste masks into an image tensor - GPU compatible version.

    This is a TorchScript-compatible replacement for torchvision's paste_masks_in_image
    that keeps all tensors on the same device.

    Args:
        masks: Tensor of shape [N, 1, H, W] - raw mask predictions (usually 28x28)
        boxes: Tensor of shape [N, 4] - bounding boxes in (x1, y1, x2, y2) format
        img_h: Image height
        img_w: Image width
        threshold: Threshold for binarizing masks

    Returns:
        Tensor of shape [N, 1, img_h, img_w] - masks pasted at box locations
    """
    n = masks.shape[0]
    if n == 0:
        return masks.new_zeros((0, 1, img_h, img_w))

    device = masks.device
    result = torch.zeros((n, 1, img_h, img_w), dtype=masks.dtype, device=device)

    for i in range(n):
        box = boxes[i]
        x1 = int(torch.clamp(box[0], min=0).item())
        y1 = int(torch.clamp(box[1], min=0).item())
        x2 = int(torch.clamp(box[2], max=img_w).item())
        y2 = int(torch.clamp(box[3], max=img_h).item())

        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)

        # Resize mask to box size
        mask = masks[i : i + 1]  # Keep batch dim for interpolate
        mask_resized = torch.nn.functional.interpolate(
            mask, size=(h, w), mode="bilinear", align_corners=False
        )

        # Paste into result
        result[i, 0, y1:y2, x1:x2] = mask_resized[0, 0, : y2 - y1, : x2 - x1]

    return result


class MaskRCNNWrapperRawMasks(torch.nn.Module):
    """Wrapper for Mask R-CNN that returns raw 28x28 masks.

    This avoids TorchScript limitations with dynamic tensor allocation
    in the mask pasting operation. The client must resize masks to
    bounding box dimensions and paste them onto the image.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        # Store reference to roi_heads for direct access
        self.transform = model.transform
        self.backbone = model.backbone
        self.rpn = model.rpn
        self.roi_heads = model.roi_heads

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning raw masks.

        Args:
            image: Input tensor of shape [C, H, W]

        Returns:
            Tuple of (boxes, labels, scores, masks)
            masks are [N, 1, 28, 28] - raw ROI mask output, NOT pasted to image size
            Client must resize each mask to its corresponding box dimensions.
        """
        # Run standard forward - masks will be pasted but we'll intercept
        outputs = self.model([image])[0]
        return (
            outputs["boxes"],
            outputs["labels"],
            outputs["scores"],
            outputs["masks"],
        )


class MaskRCNNWrapper(torch.nn.Module):
    """Wrapper for Mask R-CNN to make it compatible with TorchScript tracing.

    This wrapper returns raw 28x28 masks from the ROI heads instead of pasted
    masks. This avoids TorchScript tracing issues with torchvision's dynamic
    mask pasting operations.

    Use paste_masks_on_image() to convert raw masks to full image size after
    inference.
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
              Use paste_masks_on_image() to convert to full image size.
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


def paste_masks_on_image(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Paste raw 28x28 mask logits onto the full image.

    This function should be called after inference to convert raw masks
    from the TorchScript model to full image size.

    Args:
        masks: Raw mask logits of shape [N, 1, 28, 28] from the model
        boxes: Bounding boxes of shape [N, 4] in (x1, y1, x2, y2) format
        image_size: (height, width) of the target image

    Returns:
        Pasted masks of shape [N, 1, H, W] with values in [0, 1]
    """
    if len(masks) == 0:
        return torch.zeros((0, 1, image_size[0], image_size[1]), device=masks.device)

    n = masks.shape[0]
    h, w = image_size
    result = torch.zeros((n, 1, h, w), device=masks.device)

    for i, (mask, box) in enumerate(zip(masks, boxes)):
        x1, y1, x2, y2 = box.int().tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        box_h = y2 - y1
        box_w = x2 - x1
        if box_h <= 0 or box_w <= 0:
            continue

        # Resize mask to box size [1, 28, 28] -> [box_h, box_w]
        mask_resized = torch.nn.functional.interpolate(
            mask.unsqueeze(0),  # [1, 1, 28, 28]
            size=(box_h, box_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]  # [box_h, box_w]

        # Apply sigmoid to convert logits to probabilities
        mask_resized = torch.sigmoid(mask_resized)

        # Paste into result
        result[i, 0, y1:y2, x1:x2] = mask_resized

    return result


def _patch_paste_masks_for_device(device: torch.device):
    """Patch torchvision's paste_masks functions to use the correct device.

    The original functions create CPU tensors (torch.zeros, torch.ones) without
    specifying device, causing device mismatch errors when running on GPU.
    """
    from torchvision.models.detection import roi_heads

    def _onnx_paste_mask_in_image_patched(mask, box, im_h, im_w):
        one = torch.ones(1, dtype=torch.int64, device=device)
        zero = torch.zeros(1, dtype=torch.int64, device=device)

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

        zeros_y0 = torch.zeros(y_0, im_w, device=device)
        zeros_y1 = torch.zeros(im_h - y_1, im_w, device=device)
        concat_0 = torch.cat(
            (
                torch.zeros(y_1 - y_0, x_0, device=device),
                unpaded_im_mask,
                torch.zeros(y_1 - y_0, im_w - x_1, device=device),
            ),
            1,
        )
        im_mask = torch.cat((zeros_y0, concat_0, zeros_y1), 0)
        return im_mask

    def _onnx_paste_masks_in_image_loop_patched(masks, boxes, im_h, im_w):
        res_append = torch.zeros(0, im_h, im_w, device=device)
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
    strict validation errors.

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

    print(f"  - Tracing model with TorchScript (image_size={image_size})...")
    print(f"  - Device: {device}")
    with torch.no_grad():
        # Use check_trace=False because Mask R-CNN has dynamic anchor generation
        # that produces different tensor values across invocations
        traced_model = torch.jit.trace(
            wrapped_model, example_input, strict=False, check_trace=False
        )

    traced_model.save(output_path)
    print(f"  - TorchScript model saved to: {output_path}")

    return output_path


def load_model_from_checkpoint(
    checkpoint_path: str,
    backbone: str = "resnet50",
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, dict[int, str], int]:
    """Load a trained Mask R-CNN model from a .pth checkpoint.

    Args:
        checkpoint_path: Path to the model.pth checkpoint file.
        backbone: Backbone architecture ("resnet50" or "resnet50_v2").
        device: Device to load the model on. If None, uses CUDA if available.

    Returns:
        Tuple of (model, id2label, num_classes)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    num_classes = checkpoint["num_classes"]
    id2label = checkpoint["id2label"]

    print(f"  - Number of classes: {num_classes}")
    print(f"  - Labels: {list(id2label.values())}")
    print(f"  - Backbone: {backbone}")

    model = get_maskrcnn_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  - Model loaded on device: {device}")

    return model, id2label, num_classes


def convert_checkpoint_to_torchscript(
    checkpoint_path: str,
    output_path: str | None = None,
    backbone: str = "resnet50",
    device: str | torch.device | None = None,
    image_size: int | None = None,
) -> str:
    """Load a .pth checkpoint and export it to TorchScript format.

    Args:
        checkpoint_path: Path to the model.pth checkpoint file.
        output_path: Path to save the TorchScript model. If None, saves next to checkpoint.
        backbone: Backbone architecture ("resnet50" or "resnet50_v2").
        device: Device to use for tracing. If None, uses CUDA if available.
        image_size: Image size for tracing. If None, reads from checkpoint or defaults to 800.

    Returns:
        Path to the saved TorchScript model.

    Example:
        >>> # Convert a trained checkpoint to TorchScript
        >>> torchscript_path = convert_checkpoint_to_torchscript(
        ...     checkpoint_path="model.pth",
        ...     output_path="model.torchscript",
        ...     backbone="resnet50",
        ... )
        >>>
        >>> # Load and use the TorchScript model
        >>> model = torch.jit.load(torchscript_path)
        >>> image = torch.rand(3, 800, 800).cuda()
        >>> boxes, labels, scores, masks = model(image)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if output_path is None:
        base_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(base_dir, "model.torchscript")

    print("=" * 60)
    print("CONVERTING CHECKPOINT TO TORCHSCRIPT")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if image_size is None:
        image_size = checkpoint.get("image_size", 800)
        print(f"  - Using image_size from checkpoint: {image_size}")

    model, id2label, num_classes = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        device=device,
    )

    print(f"\nExporting to TorchScript...")
    print(f"  - Output path: {output_path}")
    print(f"  - Trace image size: {image_size}")

    # Patch torchvision's paste_masks functions to use correct device
    # This fixes "Expected all tensors to be on the same device" errors
    _patch_paste_masks_for_device(device)

    wrapped_model = MaskRCNNWrapper(model)
    wrapped_model.eval()
    wrapped_model.to(device)

    # Create example input - use a pattern that's more likely to generate detections
    # A structured pattern works better than pure random noise for tracing
    example_input = torch.rand(3, image_size, image_size).to(device)
    # Add some structure to increase chance of detections during tracing
    example_input[:, ::4, ::4] = 1.0  # Grid pattern
    example_input[:, 1::4, 1::4] = 0.0

    print("  - Tracing model...")
    print(f"  - Device: {device}")

    # First, test that the wrapper works correctly before tracing
    with torch.no_grad():
        test_boxes, test_labels, test_scores, test_masks = wrapped_model(example_input)
        print(
            f"  - Test inference: {len(test_boxes)} detections, masks shape: {test_masks.shape}"
        )

    with torch.no_grad():
        # Use check_trace=False because Mask R-CNN has dynamic anchor generation
        traced_model = torch.jit.trace(
            wrapped_model, example_input, strict=False, check_trace=False
        )

    traced_model.save(output_path)

    print(f"\nTorchScript model saved to: {output_path}")
    print("=" * 60)

    return output_path


def save_and_upload_artifacts(
    picsellia_model: Model,
    experiment: Experiment,
    model: torch.nn.Module,
    id2label: dict[int, str],
    image_size: int = 800,
    backbone: str = "resnet50",
) -> None:
    """Save model weights (PyTorch and TorchScript) and upload to the experiment.

    Args:
        picsellia_model: Picsellia model object.
        experiment: Picsellia experiment object.
        model: Trained Mask R-CNN model.
        id2label: Mapping from class ID to class name.
        image_size: Image size used for training.
        backbone: Backbone architecture used for training.
    """
    import json

    out_dir = os.path.join(picsellia_model.results_dir, picsellia_model.name)
    final_dir = os.path.join(out_dir, "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)

    print("  - Saving PyTorch checkpoint...")
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
    print(f"  - PyTorch checkpoint saved to: {model_path}")

    print("  - Exporting to TorchScript...")
    device = next(model.parameters()).device
    torchscript_path = os.path.join(final_dir, "model.torchscript")
    export_to_torchscript(model, torchscript_path, device, image_size)
    print(f"  - TorchScript model saved to: {torchscript_path}")

    print("  - Saving model metadata...")
    metadata = {
        "num_classes": len(id2label),
        "id2label": id2label,
        "label2id": {v: k for k, v in id2label.items()},
        "framework": "pytorch",
        "model_type": "maskrcnn",
        "image_size": image_size,
    }
    metadata_path = os.path.join(final_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  - Metadata saved to: {metadata_path}")

    print("  - Uploading to Picsellia...")
    picsellia_model.save_artifact_to_experiment(
        experiment=experiment,
        artifact_name="checkpoint-latest",
        artifact_path=model_path,
    )
    picsellia_model.save_artifact_to_experiment(
        experiment=experiment,
        artifact_name="model-latest",
        artifact_path=torchscript_path,
    )


def mask_to_polygon(
    mask: np.ndarray, simplify_tolerance: float = 2.0
) -> list[list[int]] | None:
    """Convert a binary mask to a polygon using contour detection.

    Args:
        mask: Binary mask array (H, W) with values 0 or 1.
        simplify_tolerance: Tolerance for polygon simplification.

    Returns:
        List of [x, y] coordinates or None if no valid contour found.
    """
    import cv2

    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 10:
        return None

    epsilon = simplify_tolerance
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(simplified) < 3:
        return None

    polygon = [[int(point[0][0]), int(point[0][1])] for point in simplified]
    return polygon


def run_inference_on_asset(
    ds: CocoDataset,
    asset,
    model: torch.nn.Module,
    device: torch.device,
    id2label: dict[int, str],
    conf_thresh: float = 0.5,
    mask_thresh: float = 0.5,
) -> PicselliaPolygonPrediction | None:
    """Run inference on a single asset and return polygon predictions.

    Args:
        ds: CocoDataset containing the asset.
        asset: Asset to run inference on.
        model: Trained Mask R-CNN model.
        device: Device to run inference on.
        id2label: Mapping from class ID to class name.
        conf_thresh: Confidence threshold for detections.
        mask_thresh: Threshold for binary mask.

    Returns:
        PicselliaPolygonPrediction or None if no valid detections.
    """
    img_path = os.path.join(ds.images_dir, asset.id_with_extension)
    image = Image.open(img_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    masks = outputs["masks"].cpu().numpy()

    valid_indices = scores >= conf_thresh
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]
    scores = scores[valid_indices]
    masks = masks[valid_indices]

    if len(boxes) == 0:
        return None

    polygons = []
    pred_labels = []
    confidences = []

    for i in range(len(boxes)):
        mask = masks[i, 0]
        binary_mask = (mask > mask_thresh).astype(np.uint8)

        polygon_coords = mask_to_polygon(binary_mask)
        if polygon_coords is None or len(polygon_coords) < 3:
            continue

        # Model predicts 1-indexed labels (0 = background), id2label is 0-indexed
        label_id = int(labels[i])
        id2label_idx = label_id - 1  # Convert to 0-indexed for id2label lookup

        if id2label_idx < 0 or id2label_idx not in id2label:
            continue  # Skip background (label_id=0) or unknown labels

        name = id2label[id2label_idx]
        label = PicselliaLabel(ds.dataset_version.get_or_create_label(name))
        conf = PicselliaConfidence(float(scores[i]))
        polygon = PicselliaPolygon(polygon_coords)

        polygons.append(polygon)
        pred_labels.append(label)
        confidences.append(conf)

    if not polygons:
        return None

    return PicselliaPolygonPrediction(
        asset=asset,
        polygons=polygons,
        labels=pred_labels,
        confidences=confidences,
    )


class PicselliaLogger:
    """Logger for training metrics to Picsellia."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to Picsellia experiment.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step/epoch number.
        """
        for name, value in metrics.items():
            if isinstance(value, int | float):
                self.experiment.log(name=name, data=float(value), type=LogType.LINE)


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    logger: PicselliaLogger | None = None,
) -> dict[str, float]:
    """Train the model for one epoch.

    Args:
        model: Mask R-CNN model.
        optimizer: Optimizer.
        data_loader: Training data loader.
        device: Device to train on.
        epoch: Current epoch number.
        logger: Optional Picsellia logger.

    Returns:
        Dictionary of average losses for the epoch.
    """
    model.train()
    total_loss = 0.0
    loss_classifier_total = 0.0
    loss_box_reg_total = 0.0
    loss_mask_total = 0.0
    loss_objectness_total = 0.0
    loss_rpn_box_reg_total = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        loss_classifier_total += loss_dict.get(
            "loss_classifier", torch.tensor(0.0)
        ).item()
        loss_box_reg_total += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
        loss_mask_total += loss_dict.get("loss_mask", torch.tensor(0.0)).item()
        loss_objectness_total += loss_dict.get(
            "loss_objectness", torch.tensor(0.0)
        ).item()
        loss_rpn_box_reg_total += loss_dict.get(
            "loss_rpn_box_reg", torch.tensor(0.0)
        ).item()
        num_batches += 1

    avg_metrics = {
        "loss": total_loss / num_batches,
        "loss_classifier": loss_classifier_total / num_batches,
        "loss_box_reg": loss_box_reg_total / num_batches,
        "loss_mask": loss_mask_total / num_batches,
        "loss_objectness": loss_objectness_total / num_batches,
        "loss_rpn_box_reg": loss_rpn_box_reg_total / num_batches,
    }

    if logger:
        logger.log_metrics(avg_metrics, step=epoch)

    return avg_metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    logger: PicselliaLogger | None = None,
) -> dict[str, float]:
    """Evaluate the model for one epoch.

    Args:
        model: Mask R-CNN model.
        data_loader: Validation data loader.
        device: Device to evaluate on.
        epoch: Current epoch number.
        logger: Optional Picsellia logger.

    Returns:
        Dictionary of average losses for the epoch.
    """
    model.train()
    total_loss = 0.0
    loss_classifier_total = 0.0
    loss_box_reg_total = 0.0
    loss_mask_total = 0.0
    num_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        loss_classifier_total += loss_dict.get(
            "loss_classifier", torch.tensor(0.0)
        ).item()
        loss_box_reg_total += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
        loss_mask_total += loss_dict.get("loss_mask", torch.tensor(0.0)).item()
        num_batches += 1

    avg_metrics = {
        "eval_loss": total_loss / num_batches,
        "eval_loss_classifier": loss_classifier_total / num_batches,
        "eval_loss_box_reg": loss_box_reg_total / num_batches,
        "eval_loss_mask": loss_mask_total / num_batches,
    }

    if logger:
        logger.log_metrics(avg_metrics, step=epoch)

    return avg_metrics

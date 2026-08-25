from __future__ import annotations

import os
from typing import Any, cast

import cv2
import numpy as np
import torch
from picsellia_cv_engine.core.data.dataset.coco_dataset import CocoDataset
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid
from transformers import Sam3Model, Sam3Processor


def _load_image_rgb(input_path: str) -> Image.Image | None:
    try:
        return Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Unable to read {input_path}. Skipping. Error: {e}")
        return None


def _coco_bbox_to_xyxy(bbox: list[float]) -> list[int]:
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def _prepare_image_prompt_context(
    *,
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    image_pil: Image.Image,
    device: str,
):
    """
    Run the vision encoder (and the constant "visual" text encoding) ONCE per
    image. The vision backbone is by far the most expensive part of a SAM-3
    forward pass, so every box prompt for this image can then reuse the same
    embeddings and only pay for the lightweight geometry/decoder heads,
    instead of re-encoding the whole image for every single box.
    """
    image_inputs = sam3_processor(images=image_pil, return_tensors="pt").to(device)
    text_inputs = sam3_processor(text="visual", return_tensors="pt").to(device)

    with torch.no_grad():
        vision_embeds = sam3_model.get_vision_features(
            pixel_values=image_inputs["pixel_values"]
        )
        # NOTE: passed back into `forward(text_embeds=...)` as-is (the model
        # reads `.pooler_output` off of it internally) - do NOT unwrap it here.
        text_embeds = sam3_model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs.get("attention_mask"),
        )

    return vision_embeds, text_embeds, image_inputs["original_sizes"]


def _infer_mask_for_box(
    *,
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    vision_embeds: Any,
    text_embeds: Any,
    original_sizes: torch.Tensor,
    device: str,
    threshold: float,
    mask_threshold: float,
    box_xyxy: list[int],
) -> np.ndarray:
    box_inputs = sam3_processor(
        original_sizes=original_sizes,
        input_boxes=[[box_xyxy]],
        input_boxes_labels=[[1]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = sam3_model(
            vision_embeds=vision_embeds,
            text_embeds=text_embeds,
            input_boxes=box_inputs["input_boxes"],
            input_boxes_labels=box_inputs["input_boxes_labels"],
        )

    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=original_sizes.tolist(),
    )[0]

    masks = results["masks"]
    if len(masks) == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)

    return masks.cpu().numpy().astype(np.uint8)


def _mask_iou_with_box(mask: np.ndarray, box_xyxy: list[int]) -> float:
    x1, y1, x2, y2 = box_xyxy
    box_mask = np.zeros_like(mask)
    box_mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 1

    intersection = float(np.logical_and(mask, box_mask).sum())
    union = float(np.logical_or(mask, box_mask).sum())
    return 0.0 if union == 0 else intersection / union


def _select_best_mask(
    masks_np: np.ndarray, box_xyxy: list[int]
) -> np.ndarray | None:
    """SAM-3 may return several candidate masks for a single box prompt.
    Keep the one whose extent best matches the box that was used as a
    prompt, since that's the object the box was meant to describe."""
    if masks_np.size == 0:
        return None
    if len(masks_np) == 1:
        return masks_np[0]

    ious = [_mask_iou_with_box(mask, box_xyxy) for mask in masks_np]
    best_idx = int(np.argmax(ious))
    return masks_np[best_idx]


def _mask_to_polygon(
    mask: np.ndarray, min_area: float
) -> tuple[list[int], list[int], float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    segmentation = contour.flatten().tolist()

    points = [
        (segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)
    ]
    if len(points) < 3:
        return None

    try:
        polygon = make_valid(Polygon(points))
        if not polygon.is_valid or polygon.is_empty:
            return None
    except Exception as e:
        print(f"   ⚠️ Error creating polygon: {e}")
        return None

    return [int(x), int(y), int(w), int(h)], segmentation, area


def _box_to_polygon_fallback(
    box_xyxy: list[int],
) -> tuple[list[int], list[int], float]:
    """Rectangular polygon matching the original box, used when SAM-3 can't
    produce a valid mask for it."""
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    segmentation = [x1, y1, x2, y1, x2, y2, x1, y2]
    return [int(x1), int(y1), int(w), int(h)], segmentation, float(w * h)


def process_boxes_to_polygons(
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    picsellia_dataset: CocoDataset,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    For every existing bounding box annotation in `picsellia_dataset`, prompt
    SAM-3 with that box to get a segmentation mask, then turn it into a
    polygon annotation carrying the SAME category (label) as the box.
    """
    images_dir = picsellia_dataset.images_dir
    coco = picsellia_dataset.coco_data or {}

    threshold = float(parameters.get("threshold", 0.3))
    mask_threshold = float(parameters.get("mask_threshold", 0.5))
    min_area = float(parameters.get("min_area", 10.0))
    fallback_to_bbox_polygon = bool(parameters.get("fallback_to_bbox_polygon", True))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_by_id = {
        int(img["id"]): img for img in cast(list[dict], coco.get("images", []))
    }
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in cast(list[dict], coco.get("annotations", [])):
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    new_annotations: list[dict[str, Any]] = []
    annotation_id = 1
    converted = 0
    fallback_count = 0
    skipped = 0

    for image_id, image_info in images_by_id.items():
        boxes_for_image = annotations_by_image.get(image_id, [])
        if not boxes_for_image:
            continue

        image_filename = cast(str, image_info["file_name"])
        input_path = os.path.join(images_dir, image_filename)

        image_pil = _load_image_rgb(input_path)
        if image_pil is None:
            skipped += len(boxes_for_image)
            continue

        print(
            f"\n🖼️  Processing image: {image_filename} "
            f"({len(boxes_for_image)} box(es))"
        )

        vision_embeds, text_embeds, original_sizes = _prepare_image_prompt_context(
            sam3_model=sam3_model,
            sam3_processor=sam3_processor,
            image_pil=image_pil,
            device=device,
        )

        for ann in boxes_for_image:
            box_xyxy = _coco_bbox_to_xyxy(ann["bbox"])
            category_id = int(ann["category_id"])

            masks_np = _infer_mask_for_box(
                sam3_model=sam3_model,
                sam3_processor=sam3_processor,
                vision_embeds=vision_embeds,
                text_embeds=text_embeds,
                original_sizes=original_sizes,
                device=device,
                threshold=threshold,
                mask_threshold=mask_threshold,
                box_xyxy=box_xyxy,
            )

            best_mask = _select_best_mask(masks_np, box_xyxy)
            result = (
                _mask_to_polygon(best_mask, min_area)
                if best_mask is not None
                else None
            )

            if result is None:
                if not fallback_to_bbox_polygon:
                    print(
                        f"   ⚠️ No mask found for annotation {ann['id']}, skipping."
                    )
                    skipped += 1
                    continue
                print(
                    f"   ↪️ No mask found for annotation {ann['id']}, "
                    "falling back to the box as a rectangular polygon."
                )
                bbox_xywh, segmentation, area = _box_to_polygon_fallback(box_xyxy)
                fallback_count += 1
            else:
                bbox_xywh, segmentation, area = result
                converted += 1

            new_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox_xywh,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco["annotations"] = new_annotations

    print(f"\n✅ Converted {converted} box(es) into polygons via SAM-3.")
    print(f"   ↪️ {fallback_count} box(es) used the rectangular fallback.")
    print(f"   ⚠️ {skipped} box(es) skipped.")

    return coco

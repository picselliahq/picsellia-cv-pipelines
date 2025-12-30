from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, cast

import cv2
import numpy as np
import torch
from picsellia_cv_engine.core.data.dataset.coco_dataset import CocoDataset
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid
from transformers import Sam3Model, Sam3Processor

from .geometry import CocoAnnotation, Detection, deduplicate_cross_class


def _parse_text_prompts(text_prompt: str | None) -> list[str]:
    if not text_prompt:
        return []

    if "," in text_prompt:
        prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
        print(f"\n{'=' * 80}")
        print(f"📋 MULTI-CLASS MODE: {len(prompts)} prompts detected")
        print(f"{'=' * 80}")
        for i, prompt in enumerate(prompts, 1):
            print(f"   {i}. '{prompt}'")
        print(f"{'=' * 80}\n")
        return prompts

    prompt = text_prompt.strip()
    print(f"\n{'=' * 80}")
    print(f"📋 SINGLE-CLASS MODE: '{prompt}'")
    print(f"{'=' * 80}\n")
    return [prompt]


def _ensure_label_and_category(
    *,
    picsellia_dataset: CocoDataset,
    coco: dict[str, Any],
    labelmap: dict[str, Any],
    prompt: str,
    next_category_id: int,
) -> int:
    label_by_name = {label.name: label for label in labelmap.values()}

    if prompt not in label_by_name:
        print(f"➕ Creating label '{prompt}' in dataset version...")
        new_label = picsellia_dataset.dataset_version.create_label(name=prompt)
        labelmap[prompt] = new_label

    coco["categories"] = coco.get("categories", [])
    categories = cast(list[dict[str, Any]], coco["categories"])
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}

    if prompt not in category_name_to_id:
        categories.append(
            cast(
                dict[str, Any],
                {"id": next_category_id, "name": prompt, "supercategory": "object"},
            )
        )
        return next_category_id + 1

    return next_category_id


def _build_category_name_to_id(coco: dict[str, Any]) -> dict[str, int]:
    coco["categories"] = coco.get("categories", [])
    categories = cast(list[dict[str, Any]], coco["categories"])
    return {cat["name"]: int(cat["id"]) for cat in categories}


def _initialize_categories_and_labels(
    *,
    picsellia_dataset: CocoDataset,
    coco: dict[str, Any],
    labelmap: dict[str, Any],
    text_prompts_list: list[str],
    label_name: str,
) -> dict[str, int]:
    coco["categories"] = coco.get("categories", [])
    category_name_to_id = _build_category_name_to_id(coco)
    next_category_id = (
        (max(category_name_to_id.values(), default=0) + 1) if category_name_to_id else 1
    )

    prompts_to_register = text_prompts_list if text_prompts_list else [label_name]

    for prompt in prompts_to_register:
        next_category_id = _ensure_label_and_category(
            picsellia_dataset=picsellia_dataset,
            coco=coco,
            labelmap=labelmap,
            prompt=prompt,
            next_category_id=next_category_id,
        )

    return _build_category_name_to_id(coco)


def _load_image_rgb(input_path: str) -> Image.Image | None:
    try:
        return Image.open(input_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Unable to read {input_path}. Skipping. Error: {e}")
        return None


def _infer_masks(
    *,
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    image_pil: Image.Image,
    device: str,
    threshold: float,
    mask_threshold: float,
    text: str | None = None,
    box_prompt: list[int] | None = None,
) -> np.ndarray:
    processor_kwargs: dict[str, Any] = {"images": image_pil, "return_tensors": "pt"}
    if text is not None:
        processor_kwargs["text"] = text
    if box_prompt is not None:
        processor_kwargs["input_boxes"] = [[box_prompt]]
        processor_kwargs["input_boxes_labels"] = [[1]]

    inputs = sam3_processor(**processor_kwargs).to(device)

    with torch.no_grad():
        outputs = sam3_model(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"]
    if len(masks) == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)

    return masks.cpu().numpy().astype(np.uint8)


def _mask_to_detection(
    *,
    mask: np.ndarray,
    min_area: float,
    category_id: int,
    prompt: str,
) -> Detection | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    bbox = [int(x), int(y), int(w), int(h)]
    segmentation = contour.flatten().tolist()

    try:
        points = [
            (segmentation[i], segmentation[i + 1])
            for i in range(0, len(segmentation), 2)
        ]
        if len(points) < 3:
            return None

        polygon = make_valid(Polygon(points))
        if not polygon.is_valid or polygon.is_empty:
            return None

        return cast(
            Detection,
            {
                "polygon": polygon,
                "category_id": category_id,
                "area": area,
                "bbox": bbox,
                "segmentation": [segmentation],
                "prompt": prompt,
            },
        )
    except Exception as e:
        print(f"   ⚠️ Error creating polygon: {e}")
        return None


def _masks_to_detections(
    masks_np: np.ndarray,
    *,
    min_area: float,
    category_id: int,
    prompt: str,
) -> list[Detection]:
    detections: list[Detection] = []
    for mask in masks_np:
        det = _mask_to_detection(
            mask=mask,
            min_area=min_area,
            category_id=category_id,
            prompt=prompt,
        )
        if det is not None:
            detections.append(det)
    return detections


def _summarize_detections(image_filename: str, detections: list[Detection]) -> None:
    print(f"\n📊 SUMMARY for {image_filename}:")
    print(f"   Total detections collected: {len(detections)}")

    if not detections:
        print("   ℹ️ No objects detected\n")
        return

    class_counts: dict[str, int] = {}
    for det in detections:
        class_counts[det["prompt"]] = class_counts.get(det["prompt"], 0) + 1

    print("   Breakdown by class:")
    for prompt, count in class_counts.items():
        print(f"      - '{prompt}': {count} detection(s)")


def _detections_to_coco_annotations(
    detections: Iterable[Detection],
    *,
    image_id: int,
    start_annotation_id: int,
) -> tuple[list[CocoAnnotation], int]:
    ann_id = start_annotation_id
    out: list[CocoAnnotation] = []

    for det in detections:
        out.append(
            cast(
                CocoAnnotation,
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": det["category_id"],
                    "segmentation": det["segmentation"],
                    "area": det["area"],
                    "bbox": det["bbox"],
                    "iscrowd": 0,
                },
            )
        )
        ann_id += 1

    return out, ann_id


def _collect_detections_for_image(
    *,
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    image_pil: Image.Image,
    image_filename: str,
    text_prompts_list: list[str],
    box_prompt: list[int] | None,
    threshold: float,
    mask_threshold: float,
    min_area: float,
    label_name: str,
    category_name_to_id: dict[str, int],
    device: str,
) -> list[Detection]:
    detections: list[Detection] = []

    if text_prompts_list:
        print(f"🔄 Running inference for {len(text_prompts_list)} prompt(s)...\n")
        for idx, prompt in enumerate(text_prompts_list, 1):
            print(
                f"   [{idx}/{len(text_prompts_list)}] Inferencing with prompt: '{prompt}'"
            )

            masks_np = _infer_masks(
                sam3_model=sam3_model,
                sam3_processor=sam3_processor,
                image_pil=image_pil,
                device=device,
                threshold=threshold,
                mask_threshold=mask_threshold,
                text=prompt,
                box_prompt=box_prompt,
            )

            if masks_np.size == 0:
                print(f"      ➜ No '{prompt}' objects detected")
                continue

            print(f"      ➜ Found {len(masks_np)} '{prompt}' mask(s)")
            detections.extend(
                _masks_to_detections(
                    masks_np,
                    min_area=min_area,
                    category_id=category_name_to_id[prompt],
                    prompt=prompt,
                )
            )

        return detections

    masks_np = _infer_masks(
        sam3_model=sam3_model,
        sam3_processor=sam3_processor,
        image_pil=image_pil,
        device=device,
        threshold=threshold,
        mask_threshold=mask_threshold,
        text=None,
        box_prompt=box_prompt,
    )

    if masks_np.size > 0:
        print(f"✓ Found {len(masks_np)} objects in {image_filename}")
        detections.extend(
            _masks_to_detections(
                masks_np,
                min_area=min_area,
                category_id=category_name_to_id[label_name],
                prompt=label_name,
            )
        )

    return detections


def _finalize_detections_for_image(
    *,
    image_filename: str,
    temp_detections: list[Detection],
    text_prompts_list: list[str],
    iou_threshold: float,
    containment_threshold: float,
    deduplication_strategy: str,
) -> list[Detection]:
    _summarize_detections(image_filename, temp_detections)

    if temp_detections and len(text_prompts_list) > 1:
        print("\n🧹 Running cross-class deduplication...")
        final_detections = cast(
            list[Detection],
            deduplicate_cross_class(
                cast(list[dict], temp_detections),
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
                strategy=deduplication_strategy,
            ),
        )
        print(f"✅ After deduplication: {len(final_detections)} final detection(s)\n")
        return final_detections

    if temp_detections:
        print("   ℹ️ Single class mode - skipping cross-class deduplication\n")

    return temp_detections


def process_images_sam3(
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    picsellia_dataset: CocoDataset,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    images_dir = picsellia_dataset.images_dir
    coco = picsellia_dataset.coco_data or {}
    labelmap = picsellia_dataset.labelmap or {}

    text_prompt = cast(str | None, parameters.get("text_prompt"))
    box_prompt = cast(list[int] | None, parameters.get("box_prompt"))
    threshold = float(parameters.get("threshold", 0.5))
    mask_threshold = float(parameters.get("mask_threshold", 0.5))
    label_name = cast(str, parameters.get("label_name", "object"))

    min_area = float(parameters.get("min_area", 100.0))
    iou_threshold = float(parameters.get("iou_threshold", 0.5))
    containment_threshold = float(parameters.get("containment_threshold", 0.8))
    deduplication_strategy = cast(
        str, parameters.get("deduplication_strategy", "keep_smaller")
    )

    if text_prompt is None and box_prompt is None:
        raise ValueError(
            "At least one of 'text_prompt' or 'box_prompt' must be provided in parameters"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_prompts_list = _parse_text_prompts(text_prompt)

    category_name_to_id = _initialize_categories_and_labels(
        picsellia_dataset=picsellia_dataset,
        coco=coco,
        labelmap=labelmap,
        text_prompts_list=text_prompts_list,
        label_name=label_name,
    )

    coco["annotations"] = coco.get("annotations", [])
    annotation_id = len(cast(list[dict[str, Any]], coco["annotations"])) + 1

    for image_info in cast(list[dict[str, Any]], coco.get("images", [])):
        image_filename = cast(str, image_info["file_name"])
        image_id = int(image_info["id"])
        input_path = os.path.join(images_dir, image_filename)

        print(f"\n{'─' * 80}")
        print(f"🖼️  Processing image: {image_filename}")
        print(f"{'─' * 80}")

        image_pil = _load_image_rgb(input_path)
        if image_pil is None:
            continue

        temp_detections = _collect_detections_for_image(
            sam3_model=sam3_model,
            sam3_processor=sam3_processor,
            image_pil=image_pil,
            image_filename=image_filename,
            text_prompts_list=text_prompts_list,
            box_prompt=box_prompt,
            threshold=threshold,
            mask_threshold=mask_threshold,
            min_area=min_area,
            label_name=label_name,
            category_name_to_id=category_name_to_id,
            device=device,
        )

        final_detections = _finalize_detections_for_image(
            image_filename=image_filename,
            temp_detections=temp_detections,
            text_prompts_list=text_prompts_list,
            iou_threshold=iou_threshold,
            containment_threshold=containment_threshold,
            deduplication_strategy=deduplication_strategy,
        )

        new_anns, annotation_id = _detections_to_coco_annotations(
            final_detections,
            image_id=image_id,
            start_annotation_id=annotation_id,
        )
        cast(list[dict[str, Any]], coco["annotations"]).extend(
            cast(list[dict[str, Any]], new_anns)
        )

    print(
        f"\n✅ Annotated {len(cast(list[dict[str, Any]], coco.get('images', [])))} images using SAM-3."
    )
    print(
        f"   📊 Total annotations: {len(cast(list[dict[str, Any]], coco.get('annotations', [])))}"
    )
    print(
        f"   📋 Categories: {len(cast(list[dict[str, Any]], coco.get('categories', [])))}"
    )

    return coco

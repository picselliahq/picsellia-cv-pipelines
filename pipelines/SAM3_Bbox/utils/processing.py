import os
from collections.abc import Iterable
from typing import Any, TypedDict, cast

import cv2
import numpy as np
import torch
from picsellia_cv_engine.core.data.dataset.coco_dataset import CocoDataset
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid
from transformers import Sam3Model, Sam3Processor

# -----------------------------
# Types
# -----------------------------


class Detection(TypedDict):
    polygon: Polygon
    category_id: int
    area: float
    bbox: list[int]
    segmentation: list[list[int]]
    prompt: str


class CocoCategory(TypedDict):
    id: int
    name: str
    supercategory: str


class CocoAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[int]]
    area: float
    bbox: list[int]
    iscrowd: int


# -----------------------------
# Geometry helpers
# -----------------------------


def calculate_iou(polygon1: Polygon, polygon2: Polygon) -> float:
    try:
        poly1 = make_valid(polygon1)
        poly2 = make_valid(polygon2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return 0.0 if union == 0 else float(intersection / union)
    except Exception as e:
        print(f"⚠️ Error calculating IoU: {e}")
        return 0.0


def _compute_overlap_metrics(det_i: Detection, det_j: Detection) -> tuple[float, float]:
    """Returns (iou, max_containment)."""
    poly_i = det_i["polygon"]
    poly_j = det_j["polygon"]

    intersection = poly_i.intersection(poly_j).area
    union = poly_i.union(poly_j).area

    iou = float(intersection / union) if union > 0 else 0.0

    area_i = det_i["area"]
    area_j = det_j["area"]

    containment_i_in_j = float(intersection / area_i) if area_i > 0 else 0.0
    containment_j_in_i = float(intersection / area_j) if area_j > 0 else 0.0
    max_containment = max(containment_i_in_j, containment_j_in_i)

    return iou, max_containment


def _find_overlaps(
    detections: list[Detection],
    iou_threshold: float,
    containment_threshold: float,
) -> list[tuple[int, int, float, float]]:
    """Build list of (i, j, iou, max_containment) for pairs that overlap enough."""
    overlaps: list[tuple[int, int, float, float]] = []
    n = len(detections)

    for i in range(n):
        for j in range(i + 1, n):
            try:
                iou, max_containment = _compute_overlap_metrics(
                    detections[i], detections[j]
                )
            except Exception as e:
                print(f"   ⚠️ Error comparing detections {i} and {j}: {e}")
                continue

            if iou > iou_threshold or max_containment > containment_threshold:
                overlaps.append((i, j, iou, max_containment))

    return overlaps


def _choose_removal(
    det_i: Detection,
    det_j: Detection,
    i: int,
    j: int,
    strategy: str,
) -> tuple[int, int]:
    """Returns (removed_idx, kept_idx)."""
    if strategy == "keep_smaller":
        return (j, i) if det_i["area"] < det_j["area"] else (i, j)

    # keep_larger
    return (j, i) if det_i["area"] > det_j["area"] else (i, j)


def _log_dedup_removal(
    removed_det: Detection,
    kept_det: Detection,
    iou: float,
    containment: float,
    iou_threshold: float,
) -> None:
    if iou > iou_threshold:
        print(
            f"   ❌ Removed '{removed_det['prompt']}' (area {removed_det['area']:.1f}) - "
            f"IoU {iou:.2f} with '{kept_det['prompt']}' (area {kept_det['area']:.1f})"
        )
    else:
        print(
            f"   ❌ Removed '{removed_det['prompt']}' (area {removed_det['area']:.1f}) - "
            f"Containment {containment:.2f} with '{kept_det['prompt']}' (area {kept_det['area']:.1f})"
        )


def deduplicate_cross_class(
    detections: list[dict],
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
    strategy: str = "keep_smaller",
) -> list[dict]:
    """
    Deduplicate masks across different classes using IoU and containment metrics.
    """
    if not detections:
        return []

    dets = cast(list[Detection], detections)

    print(f"\n🔧 Cross-class deduplication of {len(dets)} detections...")
    print(f"   - Strategy: {strategy}")
    print(f"   - IoU threshold: {iou_threshold}")
    print(f"   - Containment threshold: {containment_threshold}")

    overlaps = _find_overlaps(
        detections=dets,
        iou_threshold=iou_threshold,
        containment_threshold=containment_threshold,
    )

    to_remove: set[int] = set()

    for i, j, iou, containment in overlaps:
        if i in to_remove or j in to_remove:
            continue

        removed_idx, kept_idx = _choose_removal(
            dets[i], dets[j], i, j, strategy=strategy
        )
        to_remove.add(removed_idx)

        _log_dedup_removal(
            removed_det=dets[removed_idx],
            kept_det=dets[kept_idx],
            iou=iou,
            containment=containment,
            iou_threshold=iou_threshold,
        )

    final_detections = [det for idx, det in enumerate(dets) if idx not in to_remove]

    print(
        f"   ✓ Kept {len(final_detections)} detections, removed {len(to_remove)} duplicates\n"
    )
    return cast(list[dict], final_detections)


# -----------------------------
# Optional post-processing (kept, refactored)
# -----------------------------


def _filter_annotations_by_area(annotations: list[dict], min_area: float) -> list[dict]:
    filtered: list[dict] = []
    for ann in annotations:
        if ann["area"] >= min_area:
            filtered.append(ann)
        else:
            print(
                f"   ❌ Removed annotation {ann['id']} (area: {ann['area']:.1f} < {min_area})"
            )
    return filtered


def _segmentation_to_polygon(ann: dict) -> Polygon | None:
    try:
        segmentation = ann["segmentation"][0]
        points = [
            (segmentation[i], segmentation[i + 1])
            for i in range(0, len(segmentation), 2)
        ]

        if len(points) < 3:
            print(
                f"   ⚠️ Skipping annotation {ann['id']}: not enough points for polygon"
            )
            return None

        polygon = make_valid(Polygon(points))
        if polygon.is_valid and not polygon.is_empty:
            return polygon

        print(f"   ⚠️ Skipping annotation {ann['id']}: invalid polygon")
        return None
    except Exception as e:
        print(f"   ⚠️ Error processing annotation {ann['id']}: {e}")
        return None


def _build_polygons_with_annotations(
    annotations: list[dict],
) -> list[tuple[Polygon, dict[str, Any]]]:
    out: list[tuple[Polygon, dict[str, Any]]] = []
    for ann in annotations:
        polygon = _segmentation_to_polygon(ann)
        if polygon is not None:
            out.append((polygon, ann))
    return out


def _remove_overlapping_polygons(
    polygons_with_annotations: list[tuple[Polygon, dict[str, Any]]],
    max_overlap_ratio: float,
) -> list[tuple[Polygon, dict[str, Any]]]:
    final_items: list[tuple[Polygon, dict[str, Any]]] = []

    for poly1, ann1 in polygons_with_annotations:
        should_keep = True
        for poly2, ann2 in final_items:
            iou = calculate_iou(poly1, poly2)
            if iou > max_overlap_ratio:
                print(
                    f"   ❌ Removed annotation {ann1['id']} (overlaps {iou * 100:.1f}% with {ann2['id']})"
                )
                should_keep = False
                break
        if should_keep:
            final_items.append((poly1, ann1))

    return final_items


def post_process_annotations(
    annotations: list[dict],
    min_area: float = 100.0,
    max_overlap_ratio: float = 0.3,
) -> list[dict]:
    """
    Post-process annotations to filter by minimum area and remove overlapping polygons.
    """
    if not annotations:
        return []

    print(f"\n🔧 Post-processing {len(annotations)} annotations...")
    print(f"   - Minimum area: {min_area} pixels")
    print(f"   - Max overlap ratio: {max_overlap_ratio * 100}%")

    filtered_by_area = _filter_annotations_by_area(annotations, min_area=min_area)
    print(f"   ✓ After area filtering: {len(filtered_by_area)} annotations")
    if not filtered_by_area:
        return []

    filtered_by_area.sort(key=lambda x: x["area"], reverse=True)

    polygons_with_annotations = _build_polygons_with_annotations(filtered_by_area)

    final_polygons_with_annotations = _remove_overlapping_polygons(
        polygons_with_annotations=polygons_with_annotations,
        max_overlap_ratio=max_overlap_ratio,
    )

    print(
        f"   ✓ After overlap filtering: {len(final_polygons_with_annotations)} annotations"
    )
    print(
        f"   📊 Total removed: {len(annotations) - len(final_polygons_with_annotations)} annotations\n"
    )

    return [ann for _, ann in final_polygons_with_annotations]


# -----------------------------
# SAM-3 inference helpers
# -----------------------------


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
    """Ensure prompt exists as label + COCO category. Returns updated next_category_id."""
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


# -----------------------------
# Main entrypoint (refactored)
# -----------------------------


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

    if text_prompts_list:
        prompts_to_register = text_prompts_list
    else:
        prompts_to_register = [label_name]

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

    # box-only mode
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

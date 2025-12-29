import os
from typing import Any, List

import cv2
import numpy as np
import torch
from picsellia_cv_engine.core.data.dataset.coco_dataset import CocoDataset
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid
from transformers import Sam3Model, Sam3Processor


def calculate_iou(polygon1: Polygon, polygon2: Polygon) -> float:
    """
    Calculate Intersection over Union (IoU) between two polygons.

    Args:
        polygon1: First polygon
        polygon2: Second polygon

    Returns:
        float: IoU value between 0 and 1
    """
    try:
        # Ensure polygons are valid
        poly1 = make_valid(polygon1)
        poly2 = make_valid(polygon2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union == 0:
            return 0.0

        return intersection / union
    except Exception as e:
        print(f"⚠️ Error calculating IoU: {e}")
        return 0.0


def deduplicate_cross_class(
    detections: List[dict],
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
    strategy: str = "keep_smaller",
) -> List[dict]:
    """
    Deduplicate masks across different classes using IoU and containment metrics.

    Strategy: Keep ALL labels first, then identify and remove only truly duplicate detections
    (same object detected with multiple class labels). Only removes one of the duplicates,
    keeping the one prioritized by the strategy.

    This handles two scenarios:
    1. Similar-sized overlapping objects (detected via IoU)
    2. Small object inside/in front of large object (detected via containment)

    Args:
        detections: List of detection dictionaries with 'polygon', 'category_id', 'area', etc.
        iou_threshold: Threshold for IoU-based overlap (0-1)
        containment_threshold: Threshold for containment ratio (0-1)
        strategy: "keep_smaller" (prioritize precise masks) or "keep_larger" (prioritize complete masks)

    Returns:
        List[dict]: Deduplicated detections
    """
    if not detections:
        return []

    print(f"\n🔧 Cross-class deduplication of {len(detections)} detections...")
    print(f"   - Strategy: {strategy}")
    print(f"   - IoU threshold: {iou_threshold}")
    print(f"   - Containment threshold: {containment_threshold}")

    # Build a graph of overlapping detections
    n = len(detections)
    overlaps = []  # List of (index_i, index_j, iou, containment) tuples

    # Find all pairs of overlapping detections
    for i in range(n):
        for j in range(i + 1, n):
            poly_i = detections[i]["polygon"]
            poly_j = detections[j]["polygon"]
            area_i = detections[i]["area"]
            area_j = detections[j]["area"]

            try:
                intersection = poly_i.intersection(poly_j).area
                union = poly_i.union(poly_j).area

                # Calculate IoU
                iou = intersection / union if union > 0 else 0

                # Calculate containment ratios
                containment_i_in_j = intersection / area_i if area_i > 0 else 0
                containment_j_in_i = intersection / area_j if area_j > 0 else 0
                max_containment = max(containment_i_in_j, containment_j_in_i)

                # Check if they overlap significantly
                if iou > iou_threshold or max_containment > containment_threshold:
                    overlaps.append((i, j, iou, max_containment))

            except Exception as e:
                print(f"   ⚠️ Error comparing detections {i} and {j}: {e}")
                continue

    # Now decide which ones to remove from overlapping pairs
    to_remove = set()

    for i, j, iou, containment in overlaps:
        # Skip if either has already been marked for removal
        if i in to_remove or j in to_remove:
            continue

        det_i = detections[i]
        det_j = detections[j]

        # Apply strategy to decide which to keep
        if strategy == "keep_smaller":
            # Remove the larger one
            if det_i["area"] < det_j["area"]:
                to_remove.add(j)
                removed_idx, kept_idx = j, i
            else:
                to_remove.add(i)
                removed_idx, kept_idx = i, j
        else:  # keep_larger
            # Remove the smaller one
            if det_i["area"] > det_j["area"]:
                to_remove.add(j)
                removed_idx, kept_idx = j, i
            else:
                to_remove.add(i)
                removed_idx, kept_idx = i, j

        removed_det = detections[removed_idx]
        kept_det = detections[kept_idx]

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

    # Keep only detections not marked for removal
    final_detections = [
        det for idx, det in enumerate(detections) if idx not in to_remove
    ]

    print(
        f"   ✓ Kept {len(final_detections)} detections, removed {len(to_remove)} duplicates\n"
    )

    return final_detections


def post_process_annotations(
    annotations: List[dict], min_area: float = 100.0, max_overlap_ratio: float = 0.3
) -> List[dict]:
    """
    Post-process annotations to filter by minimum area and remove overlapping polygons.

    Args:
        annotations: List of COCO annotation dictionaries
        min_area: Minimum area threshold in pixels (default: 100.0)
        max_overlap_ratio: Maximum allowed overlap ratio between polygons (default: 0.3)

    Returns:
        List[dict]: Filtered annotations
    """
    if not annotations:
        return []

    print(f"\n🔧 Post-processing {len(annotations)} annotations...")
    print(f"   - Minimum area: {min_area} pixels")
    print(f"   - Max overlap ratio: {max_overlap_ratio * 100}%")

    # Step 1: Filter by minimum area
    filtered_by_area = []
    for ann in annotations:
        if ann["area"] >= min_area:
            filtered_by_area.append(ann)
        else:
            print(
                f"   ❌ Removed annotation {ann['id']} (area: {ann['area']:.1f} < {min_area})"
            )

    print(f"   ✓ After area filtering: {len(filtered_by_area)} annotations")

    if not filtered_by_area:
        return []

    # Step 2: Remove overlapping polygons
    # Sort by area (descending) to keep larger polygons when overlap is detected
    filtered_by_area.sort(key=lambda x: x["area"], reverse=True)

    # Convert segmentations to Shapely polygons
    polygons_with_annotations = []
    for ann in filtered_by_area:
        try:
            # COCO segmentation format: [[x1, y1, x2, y2, ..., xn, yn]]
            segmentation = ann["segmentation"][0]
            # Convert flat list to list of (x, y) tuples
            points = [
                (segmentation[i], segmentation[i + 1])
                for i in range(0, len(segmentation), 2)
            ]

            if len(points) < 3:
                print(
                    f"   ⚠️ Skipping annotation {ann['id']}: not enough points for polygon"
                )
                continue

            polygon = Polygon(points)
            polygon = make_valid(polygon)

            if polygon.is_valid and not polygon.is_empty:
                polygons_with_annotations.append((polygon, ann))
            else:
                print(f"   ⚠️ Skipping annotation {ann['id']}: invalid polygon")
        except Exception as e:
            print(f"   ⚠️ Error processing annotation {ann['id']}: {e}")
            continue

    # Keep track of which annotations to keep
    final_annotations = []
    removed_count = 0

    for i, (poly1, ann1) in enumerate(polygons_with_annotations):
        should_keep = True

        # Check overlap with all previously accepted polygons
        for poly2, ann2 in final_annotations:
            iou = calculate_iou(poly1, poly2)

            if iou > max_overlap_ratio:
                # This polygon overlaps too much with an already accepted one
                # Since we sorted by area, the already accepted one is larger
                print(
                    f"   ❌ Removed annotation {ann1['id']} (overlaps {iou * 100:.1f}% with {ann2['id']})"
                )
                should_keep = False
                removed_count += 1
                break

        if should_keep:
            final_annotations.append((poly1, ann1))

    print(f"   ✓ After overlap filtering: {len(final_annotations)} annotations")
    print(
        f"   📊 Total removed: {len(annotations) - len(final_annotations)} annotations\n"
    )

    # Extract just the annotations (without polygons)
    return [ann for _, ann in final_annotations]


def process_images_sam3(
    sam3_model: Sam3Model,
    sam3_processor: Sam3Processor,
    picsellia_dataset: CocoDataset,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Annotate a dataset using SAM-3 segmentation model.

    Supports multi-class detection via comma-separated text prompts.
    When multiple prompts are provided, runs inference for each class separately
    and deduplicates overlapping detections.

    Args:
        sam3_model (Sam3Model): SAM-3 model for segmentation.
        sam3_processor (Sam3Processor): SAM-3 processor for input preparation.
        picsellia_dataset (CocoDataset): Dataset object containing image dir, labelmap, coco metadata.
        parameters (dict[str, Any]): Parameters including 'text_prompt', 'threshold', 'mask_threshold', 'box_prompt', etc.

    Returns:
        dict[str, Any]: COCO annotations with added segmentation masks and bounding boxes.
    """
    images_dir = picsellia_dataset.images_dir
    coco = picsellia_dataset.coco_data or {}
    labelmap = picsellia_dataset.labelmap or {}

    # Get parameters
    text_prompt = parameters.get("text_prompt", None)
    box_prompt = parameters.get("box_prompt", None)
    threshold = parameters.get("threshold", 0.5)
    mask_threshold = parameters.get("mask_threshold", 0.5)
    label_name = parameters.get("label_name", "object")

    # Post-processing parameters
    min_area = parameters.get("min_area", 100.0)
    max_overlap_ratio = parameters.get("max_overlap_ratio", 0.3)

    # Multi-class deduplication parameters
    iou_threshold = parameters.get("iou_threshold", 0.5)
    containment_threshold = parameters.get("containment_threshold", 0.8)
    deduplication_strategy = parameters.get("deduplication_strategy", "keep_smaller")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if text_prompt is None and box_prompt is None:
        raise ValueError(
            "At least one of 'text_prompt' or 'box_prompt' must be provided in parameters"
        )

    # Parse text prompts - support comma-separated values for multi-class
    text_prompts_list = []
    if text_prompt:
        if "," in text_prompt:
            text_prompts_list = [p.strip() for p in text_prompt.split(",")]
            print(f"\n{'=' * 80}")
            print(f"📋 MULTI-CLASS MODE: {len(text_prompts_list)} prompts detected")
            print(f"{'=' * 80}")
            for i, prompt in enumerate(text_prompts_list, 1):
                print(f"   {i}. '{prompt}'")
            print(f"{'=' * 80}\n")
        else:
            text_prompts_list = [text_prompt]
            print(f"\n{'=' * 80}")
            print(f"📋 SINGLE-CLASS MODE: '{text_prompt}'")
            print(f"{'=' * 80}\n")

    # Build name -> Label
    label_by_name = {label.name: label for label in labelmap.values()}

    # Build name -> category_id for COCO (starting from 1)
    coco["categories"] = coco.get("categories", [])
    category_name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    next_category_id = max(category_name_to_id.values(), default=0) + 1

    # Create categories for all text prompts upfront
    if text_prompts_list:
        for prompt in text_prompts_list:
            # Use prompt as label name directly
            if prompt not in label_by_name:
                print(f"➕ Creating label '{prompt}' in dataset version...")
                new_label = picsellia_dataset.dataset_version.create_label(name=prompt)
                label_by_name[prompt] = new_label
                labelmap[prompt] = new_label

            # Ensure category_id exists for COCO
            if prompt not in category_name_to_id:
                category_name_to_id[prompt] = next_category_id
                coco["categories"].append(
                    {
                        "id": next_category_id,
                        "name": prompt,
                        "supercategory": "object",
                    }
                )
                next_category_id += 1
    else:
        # Fallback to label_name for box_prompt only scenarios
        if label_name not in label_by_name:
            print(f"➕ Creating label '{label_name}' in dataset version...")
            new_label = picsellia_dataset.dataset_version.create_label(name=label_name)
            label_by_name[label_name] = new_label
            labelmap[label_name] = new_label

        if label_name not in category_name_to_id:
            category_name_to_id[label_name] = next_category_id
            coco["categories"].append(
                {
                    "id": next_category_id,
                    "name": label_name,
                    "supercategory": "object",
                }
            )
            next_category_id += 1

    coco["annotations"] = coco.get("annotations", [])
    annotation_id = len(coco["annotations"]) + 1

    for image_info in coco["images"]:
        image_filename = image_info["file_name"]
        image_id = image_info["id"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        input_path = os.path.join(images_dir, image_filename)

        print(f"\n{'─' * 80}")
        print(f"🖼️  Processing image: {image_filename}")
        print(f"{'─' * 80}")

        # Read image with PIL for SAM-3
        try:
            image_pil = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Unable to read {input_path}. Skipping. Error: {e}")
            continue

        # Collect all detections for this image (across all prompts)
        temp_detections = []

        # Run inference for each text prompt
        if text_prompts_list:
            print(f"🔄 Running inference for {len(text_prompts_list)} prompt(s)...\n")
            for idx, single_prompt in enumerate(text_prompts_list, 1):
                print(
                    f"   [{idx}/{len(text_prompts_list)}] Inferencing with prompt: '{single_prompt}'"
                )
                # Prepare processor inputs for this specific prompt
                processor_kwargs = {
                    "images": image_pil,
                    "text": single_prompt,
                    "return_tensors": "pt",
                }

                if box_prompt is not None:
                    # Box format: [x1, y1, x2, y2] in absolute pixels
                    processor_kwargs["input_boxes"] = [[box_prompt]]
                    processor_kwargs["input_boxes_labels"] = [[1]]  # 1 = positive box

                # Process inputs
                inputs = sam3_processor(**processor_kwargs).to(device)

                # Run inference
                with torch.no_grad():
                    outputs = sam3_model(**inputs)

                # Post-process results
                results = sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                masks = results["masks"]

                if len(masks) == 0:
                    print(f"      ➜ No '{single_prompt}' objects detected")
                    continue

                print(f"      ➜ Found {len(masks)} '{single_prompt}' mask(s)")

                # Convert masks to detection format
                masks_np = masks.cpu().numpy().astype(np.uint8)

                for mask in masks_np:
                    # Find contours
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Skip if no contours found
                    if len(contours) == 0:
                        continue

                    # Get the largest contour
                    contour = max(contours, key=cv2.contourArea)

                    # Calculate area
                    area = float(cv2.contourArea(contour))

                    # Skip very small masks
                    if area < min_area:
                        continue

                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [int(x), int(y), int(w), int(h)]

                    # Convert contour to segmentation format (polygon)
                    segmentation = contour.flatten().tolist()

                    # Create polygon for deduplication
                    try:
                        points = [
                            (segmentation[i], segmentation[i + 1])
                            for i in range(0, len(segmentation), 2)
                        ]
                        if len(points) >= 3:
                            polygon = make_valid(Polygon(points))
                            if polygon.is_valid and not polygon.is_empty:
                                detection = {
                                    "polygon": polygon,
                                    "category_id": category_name_to_id[single_prompt],
                                    "area": area,
                                    "bbox": bbox,
                                    "segmentation": [segmentation],
                                    "prompt": single_prompt,
                                }
                                temp_detections.append(detection)

                    except Exception as e:
                        print(f"   ⚠️ Error creating polygon: {e}")
                        continue
        else:
            # Box prompt only (no text prompts) - original single-class behavior
            processor_kwargs = {"images": image_pil, "return_tensors": "pt"}

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

            if len(masks) > 0:
                print(f"✓ Found {len(masks)} objects in {image_filename}")
                masks_np = masks.cpu().numpy().astype(np.uint8)

                for mask in masks_np:
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if len(contours) == 0:
                        continue

                    contour = max(contours, key=cv2.contourArea)
                    area = float(cv2.contourArea(contour))

                    if area < min_area:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = [int(x), int(y), int(w), int(h)]
                    segmentation = contour.flatten().tolist()

                    try:
                        points = [
                            (segmentation[i], segmentation[i + 1])
                            for i in range(0, len(segmentation), 2)
                        ]
                        if len(points) >= 3:
                            polygon = make_valid(Polygon(points))
                            if polygon.is_valid and not polygon.is_empty:
                                temp_detections.append(
                                    {
                                        "polygon": polygon,
                                        "category_id": category_name_to_id[label_name],
                                        "area": area,
                                        "bbox": bbox,
                                        "segmentation": [segmentation],
                                        "prompt": label_name,
                                    }
                                )
                    except Exception as e:
                        print(f"   ⚠️ Error creating polygon: {e}")
                        continue

        # Deduplicate detections across classes
        # Always run deduplication if we have multiple prompts and detections
        print(f"\n📊 SUMMARY for {image_filename}:")
        print(f"   Total detections collected: {len(temp_detections)}")

        if len(temp_detections) > 0:
            # Show breakdown by class
            class_counts = {}
            for det in temp_detections:
                prompt = det["prompt"]
                class_counts[prompt] = class_counts.get(prompt, 0) + 1

            print(f"   Breakdown by class:")
            for prompt, count in class_counts.items():
                print(f"      - '{prompt}': {count} detection(s)")

            if len(text_prompts_list) > 1:
                # Multiple prompts: run cross-class deduplication
                print(f"\n🧹 Running cross-class deduplication...")
                final_detections = deduplicate_cross_class(
                    temp_detections,
                    iou_threshold=iou_threshold,
                    containment_threshold=containment_threshold,
                    strategy=deduplication_strategy,
                )
                print(
                    f"✅ After deduplication: {len(final_detections)} final detection(s)\n"
                )
            else:
                # Single prompt: no cross-class deduplication needed
                final_detections = temp_detections
                print(f"   ℹ️ Single class mode - skipping cross-class deduplication\n")
        else:
            # No detections found
            final_detections = []
            print(f"   ℹ️ No objects detected\n")

        # Convert final detections to COCO annotations
        for detection in final_detections:
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": detection["category_id"],
                "segmentation": detection["segmentation"],
                "area": detection["area"],
                "bbox": detection["bbox"],
                "iscrowd": 0,
            }

            coco["annotations"].append(annotation)
            annotation_id += 1

    print(f"\n✅ Annotated {len(coco['images'])} images using SAM-3.")
    print(f"   📊 Total annotations: {len(coco['annotations'])}")
    print(f"   📋 Categories: {len(coco['categories'])}")

    # Note: Deduplication is already applied per-image during processing
    # The old post_process_annotations is replaced by deduplicate_cross_class

    return coco

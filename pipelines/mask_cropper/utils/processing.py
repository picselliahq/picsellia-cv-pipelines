import os
from copy import deepcopy
from glob import glob
from typing import Any

from PIL import Image
import numpy as np
import cv2


def process_images(
    detection_dataset_coco: dict[str, Any],
    input_images_dir: str,
    input_coco: dict[str, Any],
    parameters: dict[str, Any],
    output_images_dir: str,
    output_coco: dict[str, Any],
) -> dict[str, Any]:
    """
    Crop images based on polygons of a label given in parameters, mask out everything else with black, and update annotations for the cropped sub-images.
    Instead of segmentation annotations, match and add detection bounding boxes from detection_dataset_coco that overlap with each crop. Adjust bounding box coordinates to the cropped image.
    The output COCO will be a detection dataset (bounding boxes only, no segmentation).
    Only keep detection bounding boxes that overlap the original polygon (not just the crop rectangle).
    """
    import shutil

    os.makedirs(output_images_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_images_dir, "*"))
    target_label = parameters.get("mask_label")
    if not target_label:
        raise ValueError("Parameter 'mask_label' must be provided in parameters.")

    # Build category_id <-> name mapping
    cat_id_to_name = {cat["id"]: cat["name"] for cat in input_coco["categories"]}
    name_to_cat_id = {cat["name"]: cat["id"] for cat in input_coco["categories"]}
    target_label_id = name_to_cat_id.get(target_label)
    if target_label_id is None:
        raise ValueError(f"Label '{target_label}' not found in categories.")

    # Prepare output COCO structure for detection
    output_coco["images"] = []
    output_coco["annotations"] = []
    output_coco["categories"] = deepcopy(detection_dataset_coco["categories"])

    # Build image filename to image_id mapping for detection COCO
    det_filename_to_id = {
        img["file_name"]: img["id"] for img in detection_dataset_coco["images"]
    }
    det_imageid_to_anns = {}
    for ann in detection_dataset_coco["annotations"]:
        det_imageid_to_anns.setdefault(ann["image_id"], []).append(ann)

    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        input_image_id = get_image_id_by_filename(input_coco, image_filename)
        annotations = [
            annotation
            for annotation in input_coco["annotations"]
            if annotation["image_id"] == input_image_id
        ]
        # Find all polygons of the target label
        target_polygons = [
            ann
            for ann in annotations
            if ann["category_id"] == target_label_id and ann.get("segmentation")
        ]
        for ann in target_polygons:
            for seg in ann["segmentation"]:
                poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                # Create mask for the polygon
                mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [poly], color=255)
                # Find bounding rect
                x, y, w, h = cv2.boundingRect(poly)
                # Crop image and mask
                cropped_img = np.zeros((h, w, 3), dtype=np.uint8)
                cropped_mask = mask[y : y + h, x : x + w]
                for c in range(3):
                    cropped_img[..., c] = img_np[y : y + h, x : x + w, c] * (
                        cropped_mask // 255
                    )
                # Save cropped image
                subimg_filename = (
                    f"{os.path.splitext(image_filename)[0]}_{ann['id']}.png"
                )
                subimg_path = os.path.join(output_images_dir, subimg_filename)
                Image.fromarray(cropped_img).save(subimg_path)
                # Register new image in COCO
                new_image_id = len(output_coco["images"])
                output_coco["images"].append(
                    {
                        "id": new_image_id,
                        "file_name": subimg_filename,
                        "width": w,
                        "height": h,
                    }
                )
                # Now, for all detection bounding boxes, check if they overlap the polygon mask and add them to the cropped image if so
                det_image_id = det_filename_to_id.get(image_filename)
                if det_image_id is None:
                    continue  # No detection annotations for this image
                det_anns = det_imageid_to_anns.get(det_image_id, [])
                for det_ann in det_anns:
                    # det_ann["bbox"] is [x, y, w, h] in original image
                    bx, by, bw, bh = det_ann["bbox"]
                    # Calculate intersection with crop
                    crop_rect = [x, y, x + w, y + h]
                    box_rect = [bx, by, bx + bw, by + bh]
                    ix1 = max(crop_rect[0], box_rect[0])
                    iy1 = max(crop_rect[1], box_rect[1])
                    ix2 = min(crop_rect[2], box_rect[2])
                    iy2 = min(crop_rect[3], box_rect[3])
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    if iw > 0 and ih > 0:
                        # There is overlap with crop, now check overlap with polygon mask
                        # Create a mask for the detection bbox
                        det_mask = np.zeros(mask.shape, dtype=np.uint8)
                        det_box_int = [int(bx), int(by), int(bx + bw), int(by + bh)]
                        cv2.rectangle(
                            det_mask,
                            (det_box_int[0], det_box_int[1]),
                            (det_box_int[2] - 1, det_box_int[3] - 1),
                            color=255,
                            thickness=-1,
                        )
                        # Overlap with polygon mask
                        overlap_mask = cv2.bitwise_and(mask, det_mask)
                        if np.any(overlap_mask):
                            # There is overlap with the polygon
                            # Now, restrict the bbox to the crop
                            adj_bx = max(bx, x) - x
                            adj_by = max(by, y) - y
                            adj_bw = min(bx + bw, x + w) - max(bx, x)
                            adj_bh = min(by + bh, y + h) - max(by, y)
                            if adj_bw > 0 and adj_bh > 0:
                                new_ann = deepcopy(det_ann)
                                new_ann["image_id"] = new_image_id
                                new_ann["id"] = len(output_coco["annotations"])
                                new_ann["bbox"] = [adj_bx, adj_by, adj_bw, adj_bh]
                                new_ann.pop("segmentation", None)
                                output_coco["annotations"].append(new_ann)
    print(
        f"✅ Processed {len(image_paths)} images and created cropped detection sub-images (polygon-overlapping boxes only)."
    )
    return output_coco


def get_image_id_by_filename(coco_data: dict[str, Any], filename: str) -> int:
    """
    Retrieve the image ID for a given filename.

    Args:
        coco_data (dict): COCO dataset structure containing images.
        filename (str): Filename of the image.

    Returns:
        int: ID of the image.
    """
    for image in coco_data["images"]:
        if image["file_name"] == filename:
            return image["id"]
    raise ValueError(f"⚠️ Image with filename '{filename}' not found.")

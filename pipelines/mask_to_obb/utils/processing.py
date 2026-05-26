import os
import shutil
from typing import Any

import cv2
import numpy as np


def process_images(
    input_images_dir: str,
    input_coco: dict[str, Any],
    parameters: dict[str, Any],
    output_images_dir: str,
    output_coco: dict[str, Any],
) -> dict[str, Any]:
    """Copy each image and replace every polygon annotation with its oriented
    bounding box, stored as a 4-point polygon segmentation."""

    os.makedirs(output_images_dir, exist_ok=True)

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in input_coco["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    converted = 0
    skipped = 0

    for image_meta in input_coco["images"]:
        filename = image_meta["file_name"]
        src_path = os.path.join(input_images_dir, filename)
        dst_path = os.path.join(output_images_dir, filename)
        shutil.copy(src_path, dst_path)

        new_image_id = len(output_coco["images"])
        output_coco["images"].append(
            {
                "id": new_image_id,
                "file_name": filename,
                "width": image_meta["width"],
                "height": image_meta["height"],
            }
        )

        for annotation in annotations_by_image.get(image_meta["id"], []):
            obb = _compute_obb(annotation.get("segmentation"))
            if obb is None:
                skipped += 1
                continue

            output_coco["annotations"].append(
                {
                    "id": len(output_coco["annotations"]),
                    "image_id": new_image_id,
                    "category_id": annotation["category_id"],
                    "segmentation": [obb["coords"]],
                    "bbox": obb["bbox"],
                    "area": obb["area"],
                    "iscrowd": 0,
                }
            )
            converted += 1

    print(
        f"✅ Converted {converted} polygons to OBBs across "
        f"{len(input_coco['images'])} images (skipped {skipped})."
    )
    return output_coco


def _compute_obb(segmentation: Any) -> dict[str, Any] | None:
    """Compute the oriented bounding box of a COCO polygon segmentation using
    cv2.minAreaRect. Returns the 4 OBB corners flattened, plus axis-aligned
    bbox and area. Returns None for RLE / empty / degenerate inputs."""
    if not segmentation or not isinstance(segmentation, list):
        return None

    points: list[list[float]] = []
    for ring in segmentation:
        if not isinstance(ring, list) or len(ring) < 6:
            continue
        for i in range(0, len(ring) - 1, 2):
            points.append([ring[i], ring[i + 1]])

    if len(points) < 3:
        return None

    contour = np.array(points, dtype=np.float32)
    (_, (w, h), _) = rect = cv2.minAreaRect(contour)
    if w <= 0 or h <= 0:
        return None

    box = cv2.boxPoints(rect)
    xs, ys = box[:, 0], box[:, 1]
    x_min, y_min = float(xs.min()), float(ys.min())
    bbox = [x_min, y_min, float(xs.max() - x_min), float(ys.max() - y_min)]

    flat = [float(c) for pt in box for c in pt]
    return {"coords": flat, "bbox": bbox, "area": float(w * h)}

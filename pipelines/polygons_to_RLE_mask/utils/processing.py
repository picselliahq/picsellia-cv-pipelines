from typing import Any

from pycocotools import mask as mask_utils


def convert_annotations_to_rle(coco_data: dict[str, Any]) -> dict[str, Any]:
    """Replace every polygon annotation's segmentation with its COCO
    RLE-encoded mask. Images are left untouched (same assets, no re-upload)."""

    images_by_id = {image["id"]: image for image in coco_data["images"]}

    converted = 0
    skipped = 0
    new_annotations: list[dict[str, Any]] = []

    for annotation in coco_data["annotations"]:
        image_meta = images_by_id.get(annotation["image_id"])
        if image_meta is None:
            skipped += 1
            continue

        rle = _polygon_to_rle(
            annotation.get("segmentation"),
            height=image_meta["height"],
            width=image_meta["width"],
        )
        if rle is None:
            skipped += 1
            continue

        new_annotation = dict(annotation)
        new_annotation["segmentation"] = rle
        new_annotation["area"] = float(mask_utils.area(rle))
        new_annotations.append(new_annotation)
        converted += 1

    coco_data["annotations"] = new_annotations

    print(
        f"Converted {converted} polygons to RLE masks across "
        f"{len(coco_data['images'])} images (skipped {skipped})."
    )
    return coco_data


def _polygon_to_rle(
    segmentation: Any, height: int, width: int
) -> dict[str, Any] | None:
    """Encode a COCO polygon segmentation into a compressed RLE dict
    ({"size": [h, w], "counts": str}). Returns None for RLE / empty / invalid input."""
    if not segmentation or not isinstance(segmentation, list):
        return None

    polygons = [
        ring for ring in segmentation if isinstance(ring, list) and len(ring) >= 6
    ]
    if not polygons:
        return None

    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = [int(s) for s in rle["size"]]
    return rle

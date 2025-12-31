from __future__ import annotations

from typing import Any, TypedDict, cast

from shapely.geometry import Polygon
from shapely.validation import make_valid

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
    if strategy == "keep_smaller":
        return (j, i) if det_i["area"] < det_j["area"] else (i, j)

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
# Optional post-processing
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

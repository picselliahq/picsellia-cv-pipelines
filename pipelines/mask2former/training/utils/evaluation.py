"""Mask evaluation helpers: RLE encoding, COCO metrics, Picsellia evaluations, overlays.

Torch-free on purpose (numpy, Pillow, pycocotools): everything here can be unit
tested without a GPU stack.

Detections are plain dicts in COCO results format::

    {"image_id": int, "category_id": int, "score": float,
     "segmentation": {"size": [h, w], "counts": str},   # compressed RLE
     "box": [x, y, w, h]}                                # model box, optional
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Index of each summary value in COCOeval.stats.
_STATS = {
    "mAP50-95": 0,
    "mAP50": 1,
    "mAP75": 2,
    "mAP-small": 3,
    "mAP-medium": 4,
    "mAP-large": 5,
    "mAR100": 8,
}
TABLE_COLUMNS = [
    "Images",
    "Instances",
    "mAP50-95(M)",
    "mAP50(M)",
    "mAP75(M)",
    "mAR100(M)",
    "mAP50-95(B)",
    "mAP50(B)",
]
_OVERLAY_PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
]


def encode_mask_rle(binary_mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary (H, W) mask as a compressed COCO RLE.

    Returns:
        JSON-serializable RLE: ``{"size": [h, w], "counts": str}``.
    """
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return {
        "size": [int(s) for s in rle["size"]],
        "counts": rle["counts"].decode("utf-8"),
    }


# --------------------------------------------------------------------------- metrics
def _load_ground_truth(gt_coco_path: str) -> COCO:
    """Load the ground truth so that every annotation is a usable mask instance.

    RLE annotations may be flagged ``iscrowd=1``, which COCOeval treats as ignore
    regions and would silently discard every instance, so it is forced to 0. A
    missing ``area`` or ``bbox`` is recomputed from the mask.
    """
    with open(gt_coco_path) as f:
        gt_data = json.load(f)

    gt_data["annotations"] = [
        a for a in gt_data["annotations"] if a.get("segmentation")
    ]
    for ann in gt_data["annotations"]:
        ann["iscrowd"] = 0

    coco_gt = COCO()
    coco_gt.dataset = gt_data
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
    for ann in coco_gt.dataset["annotations"]:
        rle = coco_gt.annToRLE(ann)
        if "area" not in ann:
            ann["area"] = float(mask_utils.area(rle))
        bbox = ann.get("bbox")
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            ann["bbox"] = [float(v) for v in mask_utils.toBbox(rle)]
    return coco_gt


def _mean_valid(values: np.ndarray) -> float:
    """Mean of the defined entries (COCOeval marks undefined ones with -1)."""
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else 0.0


def _run_coco_eval(
    coco_gt: COCO, dets: list[dict], iou_type: str, image_ids: list[int]
) -> COCOeval | None:
    """Run COCOeval silently; None when there is nothing to evaluate."""
    if not dets:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
        coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType=iou_type)
        coco_eval.params.imgIds = list(image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_eval


def _per_class(coco_eval: COCOeval | None, cat_id: int) -> dict[str, float]:
    """AP50-95 / AP50 / AP75 / AR100 of one category."""
    zero = {"mAP50-95": 0.0, "mAP50": 0.0, "mAP75": 0.0, "mAR100": 0.0}
    if coco_eval is None or cat_id not in coco_eval.params.catIds:
        return zero
    k = list(coco_eval.params.catIds).index(cat_id)
    precision = coco_eval.eval["precision"][:, :, k, 0, 2]  # [T, R], area all, 100 dets
    recall = coco_eval.eval["recall"][:, k, 0, 2]  # [T]
    return {
        "mAP50-95": _mean_valid(precision),
        "mAP50": _mean_valid(precision[0]),
        "mAP75": _mean_valid(precision[5]),
        "mAR100": _mean_valid(recall),
    }


def evaluate_detections(
    gt_coco_path: str,
    detections: list[dict[str, Any]],
    image_ids: list[int],
) -> dict[str, Any]:
    """COCO segm (and bbox when detections carry a ``box``) metrics.

    The input detections are not modified (pycocotools mutates what it is given,
    so it works on copies).

    Returns:
        ``{"values": {name: float}, "table": {"data", "rows", "columns"}}``.
        Value names carry ``(M)`` for masks and ``(B)`` for boxes, as the engine
        names its own metrics (e.g. ``mAP50-95(M)``); the caller adds the phase
        prefix. The table has one row per class plus an ``all`` row.
    """
    coco_gt = _load_ground_truth(gt_coco_path)
    image_ids = list(image_ids)

    segm_dets = [
        {k: v for k, v in d.items() if k != "box"} for d in copy.deepcopy(detections)
    ]
    box_dets = [
        {
            "image_id": d["image_id"],
            "category_id": d["category_id"],
            "score": d["score"],
            "bbox": d["box"],
        }
        for d in detections
        if d.get("box") is not None
    ]
    segm_eval = _run_coco_eval(coco_gt, segm_dets, "segm", image_ids)
    box_eval = _run_coco_eval(coco_gt, box_dets, "bbox", image_ids)

    values: dict[str, float] = {}
    for name, idx in _STATS.items():
        values[f"{name}(M)"] = (
            max(float(segm_eval.stats[idx]), 0.0) if segm_eval is not None else 0.0
        )
    for name in ("mAP50-95", "mAP50", "mAP75", "mAR100"):
        values[f"{name}(B)"] = (
            max(float(box_eval.stats[_STATS[name]]), 0.0)
            if box_eval is not None
            else 0.0
        )

    image_id_set = set(image_ids)
    rows: list[str] = ["all"]
    data: list[list[float]] = []
    n_images_all = len(image_id_set)
    n_inst_all = 0
    per_class_rows: list[tuple[str, list[float]]] = []
    for cat_id in sorted(coco_gt.getCatIds()):
        anns = [
            a
            for a in coco_gt.loadAnns(coco_gt.getAnnIds(catIds=[cat_id]))
            if a["image_id"] in image_id_set
        ]
        n_inst = len(anns)
        n_inst_all += n_inst
        seg = _per_class(segm_eval, cat_id)
        box = _per_class(box_eval, cat_id)
        per_class_rows.append(
            (
                coco_gt.cats[cat_id]["name"],
                [
                    float(len({a["image_id"] for a in anns})),
                    float(n_inst),
                    seg["mAP50-95"],
                    seg["mAP50"],
                    seg["mAP75"],
                    seg["mAR100"],
                    box["mAP50-95"],
                    box["mAP50"],
                ],
            )
        )
    data.append(
        [
            float(n_images_all),
            float(n_inst_all),
            values["mAP50-95(M)"],
            values["mAP50(M)"],
            values["mAP75(M)"],
            values["mAR100(M)"],
            values["mAP50-95(B)"],
            values["mAP50(B)"],
        ]
    )
    for name, row in per_class_rows:
        rows.append(name)
        data.append(row)

    table = {
        "data": [[round(v, 4) for v in row] for row in data],
        "rows": rows,
        "columns": TABLE_COLUMNS,
    }
    return {"values": values, "table": table}


def write_evaluation_files(
    output_dir: str,
    detections: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, str]:
    """Write ``mask_metrics.json`` and ``mask_predictions.json``; return their paths."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "mask_metrics.json")
    predictions_path = os.path.join(output_dir, "mask_predictions.json")
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(predictions_path, "w") as f:
        json.dump(detections, f)
    return {"metrics": metrics_path, "predictions": predictions_path}


# --------------------------------------------------------------------------- platform
def push_mask_evaluations(
    experiment: Any,
    ds: Any,
    detections: list[dict[str, Any]],
    image_id_to_asset: dict[int, Any],
    cat_id_to_name: dict[int, str],
    inference_type: Any,
) -> int:
    """Send RLE mask detections to Picsellia as evaluations of the experiment.

    One evaluation per asset is added with ``experiment.add_evaluation(masks=...)``
    (the platform expects the compressed RLE ``counts`` string, as for MASK
    annotations), then ``compute_evaluations_metrics`` is run and awaited so the
    evaluation view and its metrics are ready at the end.

    Args:
        experiment: Experiment that owns the evaluations.
        ds: Evaluated dataset, used to resolve the labels.
        detections: Detections to send (already filtered by score).
        image_id_to_asset: COCO image id -> Picsellia asset.
        cat_id_to_name: COCO category id -> label name.
        inference_type: ``InferenceType.MASK``.

    Returns:
        Number of assets that received an evaluation.
    """
    labels: dict[str, Any] = {}
    masks_by_image: dict[int, list[tuple[str, Any, float]]] = {}
    for det in detections:
        name = cat_id_to_name[det["category_id"]]
        if name not in labels:
            labels[name] = ds.dataset_version.get_or_create_label(name)
        masks_by_image.setdefault(det["image_id"], []).append(
            (det["segmentation"]["counts"], labels[name], det["score"])
        )

    for image_id, masks in masks_by_image.items():
        experiment.add_evaluation(image_id_to_asset[image_id], masks=masks)

    job = experiment.compute_evaluations_metrics(inference_type=inference_type)
    job.wait_for_done(attempts=120)
    return len(masks_by_image)


# --------------------------------------------------------------------------- overlays
def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def draw_prediction_overlay(
    image_path: str,
    detections: list[dict[str, Any]],
    cat_id_to_name: dict[int, str],
    out_path: str,
    alpha: float = 0.5,
) -> str:
    """Draw the detections of one image (colored masks + ``label score``) to a PNG."""
    image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
    height, width = image.shape[:2]
    cat_ids = sorted(cat_id_to_name)
    texts: list[tuple[int, int, str, tuple[int, int, int]]] = []

    for det in sorted(detections, key=lambda d: d["score"]):
        mask = mask_utils.decode(det["segmentation"]).astype(bool)
        if mask.shape != (height, width):
            continue  # annotation of another resolution: do not draw garbage
        color = _OVERLAY_PALETTE[
            cat_ids.index(det["category_id"]) % len(_OVERLAY_PALETTE)
        ]
        image[mask] = (1 - alpha) * image[mask] + alpha * np.array(
            color, dtype=np.float32
        )
        x, y, _, _ = mask_utils.toBbox(det["segmentation"])
        texts.append(
            (
                int(x),
                int(y),
                f"{cat_id_to_name[det['category_id']]} {det['score']:.2f}",
                color,
            )
        )

    out = Image.fromarray(image.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    font = _load_font(max(12, min(height, width) // 40))
    for x, y, text, color in texts:
        draw.text(
            (x + 2, y + 2),
            text,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=color,
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.save(out_path)
    return out_path


def save_prediction_overlays(
    image_paths: dict[int, str],
    detections: list[dict[str, Any]],
    cat_id_to_name: dict[int, str],
    out_dir: str,
    max_images: int = 8,
) -> list[str]:
    """Save overlays for up to ``max_images`` images, evenly spread over the set.

    Only images that have at least one detection are candidates.
    """
    by_image: dict[int, list[dict[str, Any]]] = {}
    for det in detections:
        if det["image_id"] in image_paths:
            by_image.setdefault(det["image_id"], []).append(det)
    candidates = sorted(by_image)
    if not candidates or max_images <= 0:
        return []
    picks = sorted(
        {
            candidates[i]
            for i in np.linspace(
                0, len(candidates) - 1, min(max_images, len(candidates))
            ).astype(int)
        }
    )
    return [
        draw_prediction_overlay(
            image_paths[image_id],
            by_image[image_id],
            cat_id_to_name,
            os.path.join(out_dir, f"overlay_{n:02d}.png"),
        )
        for n, image_id in enumerate(picks, start=1)
    ]

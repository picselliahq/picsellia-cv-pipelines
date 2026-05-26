import json
import os
from typing import Any

import cv2
import numpy as np
import yaml

from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.data.dataset.dataset_collection import DatasetCollection


def prepare_obb_dataset(
    picsellia_datasets: DatasetCollection[CocoDataset],
) -> str:
    """Convert each split's COCO polygon annotations to YOLO OBB labels and
    write a ``data.yaml`` ready for Ultralytics. Returns the yaml path."""
    dataset_path = picsellia_datasets.dataset_path
    if dataset_path is None:
        raise ValueError("DatasetCollection.dataset_path is not set.")

    names = list(picsellia_datasets["train"].labelmap.keys())
    name_to_index = {name: idx for idx, name in enumerate(names)}

    for split_name, dataset in picsellia_datasets.datasets.items():
        labels_dir = os.path.join(dataset_path, "labels", split_name)
        _write_obb_labels_for_split(
            dataset=dataset,
            labels_dir=labels_dir,
            name_to_index=name_to_index,
        )

    data_yaml = {
        "path": dataset_path,
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "test": os.path.join("images", "test"),
        "nc": len(names),
        "names": names,
    }
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return data_yaml_path


def _write_obb_labels_for_split(
    dataset: CocoDataset,
    labels_dir: str,
    name_to_index: dict[str, int],
) -> None:
    coco = _load_coco_data(dataset)
    os.makedirs(labels_dir, exist_ok=True)

    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    skipped = 0

    for image_meta in coco.get("images", []):
        width = image_meta["width"]
        height = image_meta["height"]
        stem, _ = os.path.splitext(image_meta["file_name"])
        label_path = os.path.join(labels_dir, f"{stem}.txt")

        lines: list[str] = []
        for ann in annotations_by_image.get(image_meta["id"], []):
            corners = _compute_obb_corners(ann.get("segmentation"))
            if corners is None:
                skipped += 1
                continue

            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name is None or cat_name not in name_to_index:
                skipped += 1
                continue

            class_index = name_to_index[cat_name]
            normalized: list[float] = []
            for x, y in corners:
                normalized.append(_clamp01(x / width))
                normalized.append(_clamp01(y / height))

            lines.append(
                " ".join([str(class_index)] + [f"{v:.6f}" for v in normalized])
            )
            converted += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    print(
        f"✅ [{dataset.name}] Wrote {converted} OBB labels "
        f"({skipped} annotations skipped) to {labels_dir}"
    )


def _load_coco_data(dataset: CocoDataset) -> dict[str, Any]:
    if dataset.coco_data is not None:
        return dataset.coco_data
    coco_path = dataset.coco_file_path or os.path.join(
        dataset.annotations_dir or "", "coco_annotations.json"
    )
    with open(coco_path) as f:
        return json.load(f)


def _compute_obb_corners(segmentation: Any) -> list[tuple[float, float]] | None:
    """Return the 4 OBB corners of a COCO polygon segmentation, or None for
    RLE / empty / degenerate inputs."""
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
    return [(float(x), float(y)) for x, y in box]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))

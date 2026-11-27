"""Batched inference on the assets of a dataset (used by the evaluation step)."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import torch
from PIL import Image
from utils.model import InstancePredictor, instances_to_detections


def run_inference_on_assets(
    images_dir: str,
    items: list[tuple[Any, int]],
    predictor: InstancePredictor,
    class_to_cat_id: dict[int, int],
    score_threshold: float,
    batch_size: int = 4,
) -> Iterator[tuple[list[tuple[Any, int, list[dict[str, Any]]]], float]]:
    """Run the model on assets by batches.

    Args:
        images_dir: Directory that holds the images.
        items: ``(asset, coco_image_id)`` pairs to run the model on.
        predictor: The model wrapper.
        class_to_cat_id: Model class index -> COCO category id.
        score_threshold: Minimum score of the instances to keep.
        batch_size: Images per forward pass.

    Yields:
        One tuple per batch: the ``(asset, image_id, detections)`` of each image
        (see ``instances_to_detections``) and the time spent in the model and its
        post-processing for the whole batch, excluding image loading and mask encoding.
    """
    step = max(1, batch_size)
    for start in range(0, len(items), step):
        batch = items[start : start + step]
        images = [
            Image.open(os.path.join(images_dir, asset.id_with_extension)).convert("RGB")
            for asset, _ in batch
        ]
        sizes = [(image.height, image.width) for image in images]
        pixel_values = predictor.preprocess(images)

        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        predict_start = time.perf_counter()
        instances = predictor.predict(pixel_values, sizes, score_threshold)
        if predictor.device.type == "cuda":
            torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - predict_start

        results = [
            (
                asset,
                image_id,
                instances_to_detections(image_instances, image_id, class_to_cat_id),
            )
            for (asset, image_id), image_instances in zip(batch, instances, strict=True)
        ]
        yield results, predict_seconds

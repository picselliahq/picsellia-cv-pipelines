from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import Dataset


class Sam3SegmentationDataset(Dataset):
    """Builds (image, concept) samples for SAM-3 promptable fine-tuning.

    SAM-3 is text-prompted: a single forward pass segments every instance of one
    concept. Each sample therefore pairs an image with a single label name (the
    "concept") and the ground-truth instances of that label in the image.

    Predictions from SAM-3 are normalized to the *original* image frame
    (boxes are scaled by original width/height, masks are resized directly to the
    original size), so targets are built in that same frame: boxes are xyxy
    normalized to [0, 1] and masks are rasterized then resized to a fixed grid.
    """

    def __init__(
        self,
        images_dir: str,
        coco_file_path: str,
        processor: Any,
        mask_resolution: int = 288,
        include_negatives: bool = False,
    ) -> None:
        self.images_dir = images_dir
        self.processor = processor
        self.mask_resolution = mask_resolution
        self.coco = COCO(coco_file_path)

        self.cat_id_to_name: dict[int, str] = {
            cat_id: cat["name"] for cat_id, cat in self.coco.cats.items()
        }
        self.concept_names: list[str] = [
            self.cat_id_to_name[cat_id] for cat_id in sorted(self.cat_id_to_name)
        ]

        # Each sample is (image_id, concept_name). Skip images whose file is
        # missing locally so a partially-downloaded split does not crash training.
        self.samples: list[tuple[int, str]] = []
        for img_id, img in self.coco.imgs.items():
            if not os.path.isfile(os.path.join(self.images_dir, img["file_name"])):
                continue

            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            present_cat_ids = {
                ann["category_id"] for ann in self.coco.loadAnns(ann_ids)
            }

            for cat_id, name in self.cat_id_to_name.items():
                if include_negatives or cat_id in present_cat_ids:
                    self.samples.append((img_id, name))

    def __len__(self) -> int:
        return len(self.samples)

    def _rasterize_mask(
        self, ann: dict, height: int, width: int
    ) -> np.ndarray | None:
        seg = ann.get("segmentation")
        if not seg:
            return None
        try:
            if isinstance(seg, dict):  # RLE
                rle = seg
            elif isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
                rles = mask_utils.frPyObjects(seg, height, width)
                rle = mask_utils.merge(rles)
            else:  # flat polygon [x1, y1, x2, y2, ...]
                rles = mask_utils.frPyObjects([seg], height, width)
                rle = mask_utils.merge(rles)
            return mask_utils.decode(rle).astype(np.uint8)
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id, concept = self.samples[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.images_dir, info["file_name"])

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        encoding = self.processor(images=image, text=concept, return_tensors="pt")

        cat_ids = self.coco.getCatIds(catNms=[concept])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
        anns = [a for a in self.coco.loadAnns(ann_ids) if int(a.get("iscrowd", 0)) == 0]

        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            mask = self._rasterize_mask(ann, orig_h, orig_w)
            if mask is None or mask.sum() == 0:
                continue

            boxes.append(
                [
                    np.clip(x / orig_w, 0.0, 1.0),
                    np.clip(y / orig_h, 0.0, 1.0),
                    np.clip((x + w) / orig_w, 0.0, 1.0),
                    np.clip((y + h) / orig_h, 0.0, 1.0),
                ]
            )
            masks.append(
                cv2.resize(
                    mask,
                    (self.mask_resolution, self.mask_resolution),
                    interpolation=cv2.INTER_NEAREST,
                )
            )

        if boxes:
            target_boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target_masks = torch.as_tensor(np.stack(masks), dtype=torch.float32)
        else:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_masks = torch.zeros(
                (0, self.mask_resolution, self.mask_resolution), dtype=torch.float32
            )

        return {
            "pixel_values": encoding["pixel_values"][0],
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "boxes": target_boxes,
            "masks": target_masks,
            "orig_size": (orig_h, orig_w),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack model inputs into batched tensors; keep per-image targets as a list."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "targets": [{"boxes": b["boxes"], "masks": b["masks"]} for b in batch],
    }

"""Dataset of MASK (RLE) annotations in the format Mask2Former expects.

Images and masks are stretched to a fixed square ``image_size``: batches then need no
padding, and the predicted masks map back to the original image by a plain resize.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torch.utils.data import Dataset

# ImageNet statistics: the ones of the Hugging Face Mask2Former checkpoints.
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)


def normalize_image(
    image: Image.Image,
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> torch.Tensor:
    """RGB image -> normalized float tensor [3, image_size, image_size]."""
    resized = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


class RandomHorizontalFlip:
    """Flip the image and its masks horizontally with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, pixel_values: torch.Tensor, mask_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.p <= 0 or torch.rand(1).item() >= self.p:
            return pixel_values, mask_labels
        return pixel_values.flip(-1), mask_labels.flip(-1)


class Mask2FormerDataset(Dataset):
    """COCO dataset with RLE masks (MASK dataset versions).

    Each item has the keys Mask2Former's forward needs: ``pixel_values`` [3, S, S],
    ``mask_labels`` [N, S, S] (one binary mask per instance) and ``class_labels`` [N]
    (0-based class index), plus ``image_id`` and the original ``size`` (height, width).
    """

    def __init__(
        self,
        images_dir: str,
        ann_json: str,
        label2id: dict[str, int],
        image_size: int,
        image_mean: tuple[float, ...] = DEFAULT_IMAGE_MEAN,
        image_std: tuple[float, ...] = DEFAULT_IMAGE_STD,
        transforms: RandomHorizontalFlip | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.coco = COCO(ann_json)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.transforms = transforms
        self.img_ids: list[int] = [
            img_id
            for img_id, img in self.coco.imgs.items()
            if os.path.isfile(os.path.join(images_dir, img["file_name"]))
        ]
        # COCO category id -> 0-based class index (classes unknown to the model are skipped)
        self.cat_to_class = {
            cat["id"]: label2id[cat["name"]]
            for cat in self.coco.cats.values()
            if cat["name"] in label2id
        }

    def __len__(self) -> int:
        return len(self.img_ids)

    def _instances(self, img_id: int, info: dict[str, Any]) -> tuple[list, list]:
        """Resized binary masks and class indices of the RLE annotations of an image."""
        masks: list[np.ndarray] = []
        classes: list[int] = []
        size = self.image_size
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id])):
            segmentation = ann.get("segmentation")
            class_id = self.cat_to_class.get(ann["category_id"])
            if class_id is None or not isinstance(segmentation, dict):
                continue  # unknown class, or not an RLE mask
            mask = mask_utils.decode(segmentation).astype(np.uint8)
            resized = np.asarray(
                Image.fromarray(mask).resize((size, size), Image.NEAREST)
            )
            if resized.any():  # a tiny mask can vanish when the image is downscaled
                masks.append(resized)
                classes.append(class_id)
        return masks, classes

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id = self.img_ids[idx]
        info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.images_dir, info["file_name"]))
        original_size = (image.height, image.width)
        pixel_values = normalize_image(
            image, self.image_size, self.image_mean, self.image_std
        )

        masks, classes = self._instances(img_id, info)
        if masks:
            mask_labels = torch.from_numpy(np.stack(masks)).float()
        else:
            mask_labels = torch.zeros((0, self.image_size, self.image_size))
        if self.transforms is not None:
            pixel_values, mask_labels = self.transforms(pixel_values, mask_labels)

        return {
            "pixel_values": pixel_values,
            "mask_labels": mask_labels,
            "class_labels": torch.tensor(classes, dtype=torch.long),
            "image_id": img_id,
            "size": original_size,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack the images; masks and labels stay lists (their length varies per image)."""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "sizes": [item["size"] for item in batch],
    }

"""Mask2Former: loading, checkpoint format, optimizer groups and instance post-processing."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import (
    AutoImageProcessor,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from utils.dataset import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD, normalize_image
from utils.evaluation import encode_mask_rle
from utils.log import get_logger

log = get_logger()

# Name fragment of the Swin backbone parameters: they train with a smaller learning rate.
_BACKBONE_MARKER = "pixel_level_module.encoder"

# Model version file holding the pretrained Hugging Face checkpoint (zip of config.json,
# model.safetensors and preprocessor_config.json), and the hub checkpoint used otherwise.
PRETRAINED_WEIGHTS_NAME = "pretrained-weights"
DEFAULT_HF_MODEL = "facebook/mask2former-swin-tiny-coco-instance"


def build_label_maps(labelmap: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    """``id2label`` (0-based, no background: Mask2Former adds its own no-object class)."""
    id2label = dict(enumerate(labelmap.keys()))
    return id2label, {name: index for index, name in id2label.items()}


def find_checkpoint_dir(*roots: str | None) -> str | None:
    """First directory under ``roots`` holding a Hugging Face checkpoint (config + weights)."""
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for current, _dirs, files in os.walk(root):
            if "config.json" in files and (
                "model.safetensors" in files or "pytorch_model.bin" in files
            ):
                return current
    return None


def resolve_pretrained_source(
    picsellia_model: Any, experiment: Any, pretrained_model: str, work_dir: str
) -> str:
    """Where to load the pretrained weights from: a local directory or a hub identifier.

    - ``pretrained_model`` set: used as it is (hub identifier or local directory).
    - else the ``pretrained-weights`` file of the model version (no Hugging Face access
      needed), taken from what ``build_model`` downloaded or, failing that, downloaded here.
    - else the default hub checkpoint.
    """
    if pretrained_model:
        return pretrained_model
    found = find_checkpoint_dir(
        getattr(picsellia_model, "pretrained_weights_path", None),
        getattr(picsellia_model, "pretrained_weights_dir", None),
    )
    if found is None:
        try:
            from picsellia_cv_engine.core.models.model_downloader import (
                ModelDownloader,
            )

            version = experiment.get_base_model_version()
            weights_file = next(
                (f for f in version.list_files() if f.name == PRETRAINED_WEIGHTS_NAME),
                None,
            )
            if weights_file is not None:
                destination = os.path.join(work_dir, "pretrained_weights")
                path = ModelDownloader().download_and_process(weights_file, destination)
                found = find_checkpoint_dir(path, destination)
        except Exception as e:  # not fatal: the hub checkpoint is the fallback
            log.warning(
                "Could not get %s from the model version: %s",
                PRETRAINED_WEIGHTS_NAME,
                e,
            )
    if found is not None:
        log.info(
            "Pretrained weights: file %s of the model version (%s)",
            PRETRAINED_WEIGHTS_NAME,
            found,
        )
        return found
    log.warning(
        "No %s file in the model version: downloading %s from the Hugging Face hub",
        PRETRAINED_WEIGHTS_NAME,
        DEFAULT_HF_MODEL,
    )
    return DEFAULT_HF_MODEL


def load_pretrained(
    pretrained_model: str, id2label: dict[int, str]
) -> tuple[Mask2FormerForUniversalSegmentation, tuple[float, ...], tuple[float, ...]]:
    """Hugging Face checkpoint with a class head re-initialized for ``id2label``.

    Returns the model and the image mean / std its image processor was trained with.
    """
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
        ignore_mismatched_sizes=True,  # the class predictor is re-created for our classes
    )
    try:
        processor = AutoImageProcessor.from_pretrained(pretrained_model)
        mean, std = tuple(processor.image_mean), tuple(processor.image_std)
    except Exception:
        log.warning("Could not read the image processor: using ImageNet statistics")
        mean, std = DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD
    return model, mean, std


def snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy of the model weights on the CPU."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def build_param_groups(
    model: torch.nn.Module,
    learning_rate: float,
    backbone_lr_multiplier: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """AdamW groups: backbone with a smaller lr, no weight decay on biases and norms."""
    buckets: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = _BACKBONE_MARKER in name
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        buckets.setdefault((is_backbone, no_decay), []).append(parameter)
    return [
        {
            "params": params,
            "lr": learning_rate * (backbone_lr_multiplier if is_backbone else 1.0),
            "weight_decay": 0.0 if no_decay else weight_decay,
        }
        for (is_backbone, no_decay), params in buckets.items()
    ]


# --------------------------------------------------------------------------- checkpoint
def checkpoint_dict(
    model: Mask2FormerForUniversalSegmentation,
    id2label: dict[int, str],
    image_size: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    pretrained_model: str,
) -> dict[str, Any]:
    """Self-contained checkpoint: weights, config and what is needed to run the model.

    The config is stored with the weights so that the model can be rebuilt without
    network access (evaluation, serving).
    """
    return {
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "id2label": id2label,
        "image_size": image_size,
        "image_mean": list(image_mean),
        "image_std": list(image_std),
        "pretrained_model": pretrained_model,
    }


def load_model_from_checkpoint(
    checkpoint_path: str,
) -> tuple[Mask2FormerForUniversalSegmentation, dict[str, Any]]:
    """Rebuild the model of a checkpoint written by ``checkpoint_dict`` (no download)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = Mask2FormerConfig.from_dict(checkpoint["config"])
    model = Mask2FormerForUniversalSegmentation(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def build_model_metadata(
    id2label: dict[int, str],
    image_size: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    pretrained_model: str,
    score_threshold: float,
    mask_threshold: float,
) -> dict[str, Any]:
    """Describe the checkpoint so that it can be used without this repository."""
    labels = [id2label[i] for i in sorted(id2label)]
    return {
        "framework": "pytorch",
        "model_type": "mask2former",
        "task": "instance segmentation (MASK)",
        "pretrained_model": pretrained_model,
        "labels": labels,
        "id2label": {str(i): id2label[i] for i in sorted(id2label)},
        "num_classes": len(labels),
        "label_offset": 0,
        "image_size": image_size,
        "input": (
            f"RGB image stretched to {image_size}x{image_size}, scaled to [0, 1] then "
            "normalized with image_mean / image_std"
        ),
        "image_mean": list(image_mean),
        "image_std": list(image_std),
        "default_thresholds": {"score": score_threshold, "mask": mask_threshold},
        "output": (
            "Mask2FormerForUniversalSegmentation outputs: class_queries_logits "
            "[Q, num_classes + 1] (last = no object) and masks_queries_logits "
            "[Q, H/4, W/4]; use Mask2FormerImageProcessor.post_process_instance_segmentation "
            "with target_sizes to get one binary mask, label and score per instance"
        ),
        "how_to_load": (
            "config = Mask2FormerConfig.from_dict(checkpoint['config']); "
            "model = Mask2FormerForUniversalSegmentation(config); "
            "model.load_state_dict(checkpoint['model_state_dict'])"
        ),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


# --------------------------------------------------------------------------- inference
class InstancePredictor:
    """Run Mask2Former on images and turn its output into one binary mask per instance."""

    def __init__(
        self,
        model: Mask2FormerForUniversalSegmentation,
        device: torch.device,
        image_size: int,
        image_mean: tuple[float, ...],
        image_std: tuple[float, ...],
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.mask_threshold = mask_threshold
        self.overlap_mask_area_threshold = overlap_mask_area_threshold
        # Only used for post-processing: it holds no weights and needs no download.
        self._processor = Mask2FormerImageProcessor()

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack(
            [
                normalize_image(img, self.image_size, self.image_mean, self.image_std)
                for img in images
            ]
        )

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        sizes: list[tuple[int, int]],
        score_threshold: float,
    ) -> list[list[dict[str, Any]]]:
        """Per image, its instances: ``{"class_index", "score", "mask"}`` (mask uint8 H x W).

        ``sizes`` are the original (height, width) the masks are resized back to.
        """
        outputs = self.model(pixel_values=pixel_values.to(self.device))
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=self.mask_threshold,
            overlap_mask_area_threshold=self.overlap_mask_area_threshold,
            target_sizes=sizes,
            return_binary_maps=True,
        )
        instances: list[list[dict[str, Any]]] = []
        for result in results:
            binary_maps = (result["segmentation"] > 0.5).to(torch.uint8).cpu().numpy()
            per_image = []
            for mask, segment in zip(
                binary_maps, result["segments_info"], strict=False
            ):
                if mask.any():
                    per_image.append(
                        {
                            "class_index": int(segment["label_id"]),
                            "score": float(segment["score"]),
                            "mask": mask,
                        }
                    )
            instances.append(per_image)
        return instances


def instances_to_detections(
    instances: list[dict[str, Any]],
    image_id: int,
    class_to_cat_id: dict[int, int],
) -> list[dict[str, Any]]:
    """Instances of one image -> COCO-style RLE detections (see utils.evaluation).

    The box is the tight box of the mask, so the box metrics describe the masks.
    """
    detections: list[dict[str, Any]] = []
    for instance in instances:
        cat_id = class_to_cat_id.get(instance["class_index"])
        if cat_id is None:
            continue  # label absent from the ground-truth categories
        rle = encode_mask_rle(np.asarray(instance["mask"], dtype=np.uint8))
        x, y, w, h = (float(v) for v in mask_utils.toBbox(rle))
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "score": float(instance["score"]),
                "segmentation": rle,
                "box": [x, y, w, h],
            }
        )
    return detections

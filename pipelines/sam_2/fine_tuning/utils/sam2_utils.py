import os
import json
import shutil
import re
from typing import Any
from PIL import Image, ImageDraw


def prepare_directories(img_root: str, ann_root: str):
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(ann_root, exist_ok=True)


def load_coco_annotations(coco_path: str) -> dict[str, Any]:
    with open(coco_path, "r") as f:
        return json.load(f)


def generate_mask(width, height, annotations) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    object_idx = 1
    for ann in annotations:
        if ann.get("iscrowd", 0) == 1 or "segmentation" not in ann:
            continue
        for seg in ann["segmentation"]:
            if isinstance(seg, list) and len(seg) >= 6:
                poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                draw.polygon(poly, fill=object_idx)
                object_idx += 1
    return mask


def convert_coco_to_png_masks(coco, source_images, img_root, ann_root):
    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations_by_image: dict[str, Any] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, annotations in annotations_by_image.items():
        img_info = images_by_id[img_id]
        width, height = img_info["width"], img_info["height"]
        original_file = img_info["file_name"]
        base_name = os.path.splitext(original_file)[0]

        video_img_dir = os.path.join(img_root, base_name)
        video_ann_dir = os.path.join(ann_root, base_name)
        os.makedirs(video_img_dir, exist_ok=True)
        os.makedirs(video_ann_dir, exist_ok=True)

        shutil.copy(
            os.path.join(source_images, original_file),
            os.path.join(video_img_dir, "00000.jpg"),
        )

        mask = generate_mask(width, height, annotations)
        mask.save(os.path.join(video_ann_dir, "00000.png"))


def normalize_filenames(root_dirs: list[str]):
    for root in root_dirs:
        for subdir, _, files in os.walk(root):
            for name in files:
                new_name = name.replace(".", "_", name.count(".") - 1)
                if not re.search(r"_\d+\.\w+$", new_name):
                    new_name = new_name.replace(".", "_1.")
                os.rename(os.path.join(subdir, name), os.path.join(subdir, new_name))

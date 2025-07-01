import os
from typing import Any

import cv2
from groundingdino.util.inference import Model
from picsellia_cv_engine.core import Model as PicselliaModel


def process_images(
    picsellia_model: PicselliaModel,
    images_dir: str,
    coco: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    label_names = [cat["name"] for cat in coco.get("categories", [])]

    groundingdino_model = Model(
        model_config_path=picsellia_model.config_path,
        model_checkpoint_path=picsellia_model.pretrained_weights_path,
    )

    coco["annotations"] = []  # reset annotations
    for image_info in coco["images"]:
        image_filename = image_info["file_name"]
        image_id = image_info["id"]

        input_path = os.path.join(images_dir, image_filename)
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"⚠️ Unable to read {input_path}. Skipping.")
            continue

        detections = groundingdino_model.predict_with_classes(
            image=image_bgr,
            classes=label_names,
            box_threshold=parameters.get("box_threshold", 0.5),
            text_threshold=parameters.get("text_threshold", 0.5),
        )

        for ann_id, box in enumerate(detections.xyxy):
            category_id = (
                int(detections.class_id[ann_id])
                if detections.class_id[ann_id] is not None
                else None
            )
            if category_id is None:
                continue

            x_min, y_min, x_max, y_max = box
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]),
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        max(int(x_min), 0),
                        max(int(y_min), 0),
                        int(x_max - x_min),
                        int(y_max - y_min),
                    ],
                    "area": float((x_max - x_min) * (y_max - y_min)),
                    "iscrowd": 0,
                }
            )

    print(f"✅ Annotated {len(coco['images'])} images with GroundingDINO.")
    return coco


def get_image_id_by_filename(coco_data: dict[str, Any], filename: str) -> int:
    """
    Retrieve the image ID for a given filename.

    Args:
        coco_data (dict): COCO dataset structure containing images.
        filename (str): Filename of the image.

    Returns:
        int: ID of the image.
    """
    for image in coco_data["images"]:
        if image["file_name"] == filename:
            return image["id"]
    raise ValueError(f"⚠️ Image with filename '{filename}' not found.")

import os
from typing import Any

import cv2
from picsellia_cv_engine.core import Model as PicselliaModel
from picsellia_cv_engine.core.data.dataset.coco_dataset import CocoDataset
from ultralytics import YOLO


def process_images(
    picsellia_model: PicselliaModel,
    picsellia_dataset: CocoDataset,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Annotate a dataset using a YOLOv8 Ultralytics detection model.

    Args:
        picsellia_model (PicselliaModel): Contains the model paths (weights).
        picsellia_dataset (CocoDataset): Dataset object containing image dir, labelmap, coco metadata.
        parameters (dict[str, Any]): Parameters including 'threshold' and others.

    Returns:
        dict[str, Any]: COCO annotations with added bounding boxes.
    """
    images_dir = picsellia_dataset.images_dir
    coco = picsellia_dataset.coco_data or {}
    labelmap = picsellia_dataset.labelmap or {}
    threshold = parameters.get("threshold", 0.1)

    # Build name -> Label
    label_by_name = {label.name: label for label in labelmap.values()}

    # Build name -> category_id for COCO (starting from 1)
    coco["categories"] = coco.get("categories", [])
    category_name_to_id = {
        cat["name"]: cat["id"] for cat in coco["categories"]
    }
    next_category_id = max(category_name_to_id.values(), default=0) + 1

    model = YOLO(picsellia_model.pretrained_weights_path)
    coco["annotations"] = []  # Reset annotations

    for image_info in coco["images"]:
        image_filename = image_info["file_name"]
        image_id = image_info["id"]

        input_path = os.path.join(images_dir, image_filename)
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            print(f"⚠️ Unable to read {input_path}. Skipping.")
            continue

        results = model.predict(image_bgr, verbose=False)[0]

        for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
            score = float(results.boxes.conf[i].cpu().item())
            if score < threshold:
                continue

            class_index = int(results.boxes.cls[i].cpu().item())
            class_name = results.names[class_index]

            # Ensure Label exists in Picsellia
            if class_name not in label_by_name:
                print(f"➕ Creating missing label '{class_name}' in dataset version...")
                new_label = picsellia_dataset.dataset_version.create_label(name=class_name)
                label_by_name[class_name] = new_label
                labelmap[class_name] = new_label

            # Ensure category_id exists for COCO
            if class_name not in category_name_to_id:
                category_name_to_id[class_name] = next_category_id
                coco["categories"].append({
                    "id": next_category_id,
                    "name": class_name,
                    "supercategory": "",  # optional
                })
                next_category_id += 1

            category_id = category_name_to_id[class_name]

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

    print(f"✅ Annotated {len(coco['images'])} images using YOLOv8.")
    return coco

import os
from copy import deepcopy
from glob import glob
from typing import Any, Optional

import albumentations as A
import numpy as np
from picsellia.types.enums import InferenceType
from PIL import Image
from shapely.geometry import Polygon, box


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(pil_image)


def numpy_to_pil(numpy_image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray(numpy_image)


def get_augmentation_pipeline(
    parameters: dict, inference_type: InferenceType
) -> A.Compose:
    """
    Create an Albumentations augmentation pipeline based on dataset type.

    Args:
        parameters: Augmentation parameters
        inference_type: Type of dataset (classification, bbox, segmentation)

    Returns:
        A.Compose: Augmentation pipeline configured for the dataset type
    """
    rotate_min = parameters.get("rotate_min", -45)
    rotate_max = parameters.get("rotate_max", 45)
    scale_min = parameters.get("scale_min", 0.9)
    scale_max = parameters.get("scale_max", 1.1)
    rotate_prob = parameters.get("rotate_prob", 0.5)

    augmentations = [
        A.Affine(
            rotate=(rotate_min, rotate_max), scale=(scale_min, scale_max), p=rotate_prob
        )
    ]

    if parameters.get("add_noise", False):
        augmentations.append(
            A.OneOf(
                [
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ],
                p=0.3,
            )
        )

    compose_kwargs = {}

    if inference_type == InferenceType.OBJECT_DETECTION:
        compose_kwargs["bbox_params"] = A.BboxParams(
            format="coco", label_fields=["bbox_labels"]
        )
    elif inference_type == InferenceType.SEGMENTATION:
        compose_kwargs["keypoint_params"] = A.KeypointParams(
            format="xy", label_fields=["keypoint_ids"], remove_invisible=False
        )

    return A.Compose(augmentations, **compose_kwargs)


def is_valid_bbox(bbox: list[float], min_size: float = 1.0) -> bool:
    """
    Check if a bounding box is valid (has positive dimensions and minimum size).

    Args:
        bbox: Bounding box in COCO format [x, y, width, height]
        min_size: Minimum size for width and height

    Returns:
        bool: True if the bounding box is valid
    """
    x, y, w, h = bbox
    return w >= min_size and h >= min_size and x >= 0 and y >= 0


def is_valid_polygon(polygon: list[float]) -> bool:
    """
    Check if a polygon is valid (has minimum size and all points are valid).

    Args:
        polygon: list of polygon points [x1, y1, x2, y2, ...]
        min_size: Minimum size for width and height

    Returns:
        bool: True if the polygon is valid
    """
    if len(polygon) < 6:  # Need at least 3 points (6 coordinates) for a polygon
        return False

    # Convert to numpy array for easier manipulation
    points = np.array(polygon).reshape(-1, 2)

    # Check if all points are within valid range
    if np.any(points < 0):
        return False

    return True


def apply_classification_augmentation(
    img: Image.Image, parameters: dict[str, Any]
) -> Image.Image:
    """
    Apply augmentation to classification images (no annotations needed).

    Args:
        img: Input PIL image
        parameters: Augmentation parameters

    Returns:
        Image.Image: Augmented image
    """
    img_np = pil_to_numpy(img)
    transform = get_augmentation_pipeline(parameters, InferenceType.CLASSIFICATION)

    transformed = transform(image=img_np)
    return numpy_to_pil(transformed["image"])


def apply_bbox_augmentation(
    img: Image.Image, annotations: list[dict[str, Any]], parameters: dict[str, Any]
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """
    Apply augmentation to images with bounding box annotations.

    Args:
        img: Input PIL image
        annotations: list of COCO annotations with bboxes
        parameters: Augmentation parameters

    Returns:
        tuple of (augmented_image, transformed_annotations)
    """
    img_np = pil_to_numpy(img)
    transform = get_augmentation_pipeline(parameters, InferenceType.OBJECT_DETECTION)

    # Prepare bboxes for transformation
    bboxes = []
    bbox_labels = []

    for ann in annotations:
        if "bbox" in ann:
            bboxes.append(ann["bbox"])
            bbox_labels.append(ann["category_id"])

    # Apply transformation
    transformed = transform(image=img_np, bboxes=bboxes, bbox_labels=bbox_labels)

    processed_img = numpy_to_pil(transformed["image"])
    transformed_annotations = []

    # Reconstruct annotations
    for bbox, category_id in zip(transformed["bboxes"], transformed["bbox_labels"]):
        if is_valid_bbox(bbox):
            transformed_annotations.append(
                {
                    "bbox": bbox,
                    "category_id": category_id,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                    "id": -1,  # Will be reassigned later
                }
            )

    return processed_img, transformed_annotations


def prepare_keypoints_and_map(
    annotations: list[dict[str, Any]], img_width: int, img_height: int
) -> tuple[list[tuple[float, float]], list[int], list[tuple[int, list[list[int]]]]]:
    keypoints = []
    keypoint_ids = []
    annotation_map = []

    for ann in annotations:
        if "segmentation" in ann and ann["segmentation"]:
            ann_keypoints = []
            for polygon in ann["segmentation"]:
                polygon_keypoints = []
                for i in range(0, len(polygon), 2):
                    x = max(0, min(polygon[i], img_width - 1))
                    y = max(0, min(polygon[i + 1], img_height - 1))
                    keypoints.append((x, y))
                    keypoint_ids.append(ann["id"])
                    polygon_keypoints.append(len(keypoints) - 1)
                ann_keypoints.append(polygon_keypoints)
            annotation_map.append((ann["id"], ann_keypoints))
    return keypoints, keypoint_ids, annotation_map


def clip_polygon_to_image(
    polygon_coords: list[tuple[float, float]], img_width: int, img_height: int
) -> Optional[list[float]]:
    if len(polygon_coords) < 3:
        return None  # not a polygon

    poly = Polygon(polygon_coords)

    # Fix known invalid geometries
    if not poly.is_valid:
        poly = poly.buffer(0)  # attempt to "clean" the polygon

    if poly.is_empty or not poly.is_valid or poly.geom_type != "Polygon":
        return None

    image_bounds = box(0, 0, img_width - 1, img_height - 1)

    try:
        clipped = poly.intersection(image_bounds)
    except Exception as e:
        print(f"⚠️ Shapely intersection failed: {e}")
        return None

    if not clipped.is_empty and clipped.geom_type == "Polygon":
        coords = list(clipped.exterior.coords)
        return [coord for x, y in coords for coord in (x, y)]

    return None


def apply_segmentation_augmentation(
    img: Image.Image, annotations: list[dict[str, Any]], parameters: dict[str, Any]
) -> tuple[Image.Image, list[dict[str, Any]]]:
    img_np = pil_to_numpy(img)
    transform = get_augmentation_pipeline(parameters, InferenceType.SEGMENTATION)
    img_width, img_height = img.size

    keypoints, keypoint_ids, annotation_map = prepare_keypoints_and_map(
        annotations, img_width, img_height
    )
    ann_id_to_ann = {ann["id"]: ann for ann in annotations}

    if not keypoints:
        return img, []

    try:
        transformed = transform(
            image=img_np, keypoints=keypoints, keypoint_ids=keypoint_ids
        )
    except Exception as e:
        print(f"Warning: Transformation failed: {str(e)}, returning original image")
        return img, []

    processed_img = numpy_to_pil(transformed["image"])
    transformed_annotations = []

    for ann_id, ann_keypoints in annotation_map:
        original_ann = ann_id_to_ann[ann_id]
        new_segmentation = []

        for polygon_keypoints in ann_keypoints:
            polygon_coords = []
            for kp_idx in polygon_keypoints:
                if kp_idx < len(transformed["keypoints"]):
                    x, y = transformed["keypoints"][kp_idx]
                    polygon_coords.append((x, y))

            clipped = clip_polygon_to_image(polygon_coords, img_width, img_height)
            if clipped:
                new_segmentation.append(clipped)

        if new_segmentation:
            new_ann = deepcopy(original_ann)
            new_ann["segmentation"] = new_segmentation

            if "area" not in new_ann or new_ann["area"] == 0:
                points = np.array(new_segmentation[0]).reshape(-1, 2)
                min_x, min_y = np.min(points, axis=0)
                max_x, max_y = np.max(points, axis=0)
                new_ann["area"] = (max_x - min_x) * (max_y - min_y)

            new_ann["id"] = -1
            transformed_annotations.append(new_ann)

    return processed_img, transformed_annotations


def process_images(
    input_images_dir: str,
    input_coco: Optional[dict[str, Any]],
    parameters: dict[str, Any],
    output_images_dir: str,
    output_coco: Optional[dict[str, Any]],
    inference_type: InferenceType,
) -> Optional[dict[str, Any]]:
    """
    Process images and their annotations using Albumentations augmentations.

    Args:
        input_images_dir: Directory containing input images
        input_coco: Input COCO annotations (None for classification)
        parameters: Augmentation parameters
        output_images_dir: Directory for output images
        output_coco: Output COCO structure (None for classification)
        inference_type: Type of dataset (classification, bbox, segmentation)

    Returns:
        Updated COCO structure (None for classification)
    """
    os.makedirs(output_images_dir, exist_ok=True)

    # Get all input images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_images_dir, ext)))
        image_paths.extend(glob(os.path.join(input_images_dir, ext.upper())))

    print(f"Found {len(image_paths)} images to process")

    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        # Open the image
        img = Image.open(image_path).convert("RGB")

        if inference_type == InferenceType.CLASSIFICATION:
            # Classification: only process the image
            processed_img = apply_classification_augmentation(img, parameters)
            processed_img.save(os.path.join(output_images_dir, image_filename))

        elif inference_type == InferenceType.OBJECT_DETECTION:
            # Bbox detection: process image and bounding boxes
            if input_coco is None or output_coco is None:
                raise ValueError("COCO annotations required for bbox dataset")

            input_image_id = get_image_id_by_filename(input_coco, image_filename)
            annotations = [
                ann
                for ann in input_coco["annotations"]
                if ann["image_id"] == input_image_id and "bbox" in ann
            ]

            processed_img, transformed_annotations = apply_bbox_augmentation(
                img, annotations, parameters
            )
            processed_img.save(os.path.join(output_images_dir, image_filename))

            # Update COCO structure
            new_image_id = len(output_coco["images"])
            output_coco["images"].append(
                {
                    "id": new_image_id,
                    "file_name": image_filename,
                    "width": processed_img.width,
                    "height": processed_img.height,
                }
            )

            for annotation in transformed_annotations:
                annotation["image_id"] = new_image_id
                annotation["id"] = len(output_coco["annotations"])
                output_coco["annotations"].append(annotation)

        elif inference_type == InferenceType.SEGMENTATION:
            # Segmentation: process image and polygons
            if input_coco is None or output_coco is None:
                raise ValueError("COCO annotations required for segmentation dataset")

            input_image_id = get_image_id_by_filename(input_coco, image_filename)
            annotations = [
                ann
                for ann in input_coco["annotations"]
                if ann["image_id"] == input_image_id and "segmentation" in ann
            ]

            processed_img, transformed_annotations = apply_segmentation_augmentation(
                img, annotations, parameters
            )
            processed_img.save(os.path.join(output_images_dir, image_filename))

            # Update COCO structure
            new_image_id = len(output_coco["images"])
            output_coco["images"].append(
                {
                    "id": new_image_id,
                    "file_name": image_filename,
                    "width": processed_img.width,
                    "height": processed_img.height,
                }
            )

            for annotation in transformed_annotations:
                annotation["image_id"] = new_image_id
                annotation["id"] = len(output_coco["annotations"])
                output_coco["annotations"].append(annotation)

    print(f"✅ Processed {len(image_paths)} images for {inference_type.value} dataset.")
    return output_coco


def get_image_id_by_filename(coco_data: dict[str, Any], filename: str) -> int:
    """
    Retrieve the image ID for a given filename.

    Args:
        coco_data: COCO dataset structure containing images
        filename: Filename of the image

    Returns:
        int: ID of the image

    Raises:
        ValueError: If image filename is not found
    """
    for image in coco_data["images"]:
        if image["file_name"] == filename:
            return image["id"]
    raise ValueError(f"⚠️ Image with filename '{filename}' not found.")

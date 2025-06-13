import os
from copy import deepcopy
from glob import glob
from typing import Dict, Any, List, Tuple

import albumentations as A
import numpy as np
from PIL import Image


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(pil_image)


def numpy_to_pil(numpy_image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray(numpy_image)


def get_augmentation_pipeline(parameters: dict) -> A.Compose:
    """
    Create an Albumentations augmentation pipeline.
    Returns a pipeline that can be applied to images and annotations.
    """
    rotate_min = parameters.get("rotate_min", -45)
    rotate_max = parameters.get("rotate_max", 45)
    scale_min = parameters.get("scale_min", 0.9)
    scale_max = parameters.get("scale_max", 1.1)
    rotate_prob = parameters.get("rotate_prob", 0.5)
    augmentations = [
        A.Affine(
            rotate=(rotate_min, rotate_max),
            scale=(scale_min, scale_max),
            p=rotate_prob
        )]
    if parameters["add_noise"] == True:
        augmentations.append(
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
        ],
                p=0.3))
    return A.Compose(augmentations, bbox_params=A.BboxParams(format='coco', label_fields=['labels']),
       keypoint_params=A.KeypointParams(format='xy', label_fields=['labels']))


def is_valid_bbox(bbox: List[float], min_size: float = 1.0) -> bool:
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


def is_valid_polygon(polygon: List[float], min_size: float = 1.0) -> bool:
    """
    Check if a polygon is valid (has minimum size and all points are valid).
    
    Args:
        polygon: List of polygon points [x1, y1, x2, y2, ...]
        min_size: Minimum size for width and height
        
    Returns:
        bool: True if the polygon is valid
    """
    if len(polygon) < 6:  # Need at least 3 points (6 coordinates) for a polygon
        return False
    
    # Convert to numpy array for easier manipulation
    points = np.array(polygon).reshape(-1, 2)
    
    # Check if all points are within image bounds
    if np.any(points < 0):
        return False
    
    # Calculate bounding box of polygon
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    # Check if polygon has minimum size
    return (max_x - min_x) >= min_size and (max_y - min_y) >= min_size


def transform_annotations(
    annotations: List[Dict[str, Any]],
    transform: A.Compose,
    image_height: int,
    image_width: int
) -> List[Dict[str, Any]]:
    """
    Transform annotations according to the applied image transformations.
    
    Args:
        annotations: List of COCO annotations
        transform: Albumentations transform that was applied to the image
        image_height: Height of the image
        image_width: Width of the image
        
    Returns:
        List of transformed annotations
    """
    transformed_annotations = []
    
    for ann in annotations:
        new_ann = deepcopy(ann)
        
        # Handle bounding boxes
        if 'bbox' in ann:
            bbox = ann['bbox']  # [x, y, width, height] in COCO format
            category_id = ann['category_id']
            
            # Convert COCO bbox to Albumentations format [x_min, y_min, x_max, y_max] and normalize
            x_min = bbox[0] / image_width
            y_min = bbox[1] / image_height
            x_max = (bbox[0] + bbox[2]) / image_width
            y_max = (bbox[1] + bbox[3]) / image_height
            
            bbox_alb = [x_min, y_min, x_max, y_max]
            
            # Apply transformation
            transformed = transform(
                image=np.zeros((image_height, image_width, 3)),  # Dummy image
                bboxes=[bbox_alb],
                labels=[category_id]
            )
            if transformed['bboxes']:  # If bbox is still valid after transformation
                # Convert back to COCO format [x, y, width, height] and denormalize
                x_min, y_min, x_max, y_max = transformed['bboxes'][0]
                new_bbox = [
                   int(x_min * image_width),
                    int(y_min * image_height),
                    int((x_max - x_min) * image_width),
                    int((y_max - y_min) * image_height)
                ]
                # Only add if the transformed bbox is valid
                if is_valid_bbox(new_bbox):
                    new_ann['bbox'] = new_bbox
                    transformed_annotations.append(new_ann)
        
        # Handle segmentation polygons
        elif 'segmentation' in ann:
            # Convert polygon points to keypoints format and normalize
            keypoints = []
            keypoint_labels = []
            
            for polygon in ann['segmentation']:
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / image_width
                    y = polygon[i + 1] / image_height
                    keypoints.extend([x, y])
                    keypoint_labels.append(1)  # 1 indicates a polygon point
            
            # Apply transformation
            transformed = transform(
                image=np.zeros((image_height, image_width, 3)),  # Dummy image
                keypoints=keypoints,
                labels=keypoint_labels
            )
            
            if transformed['keypoints']:  # If keypoints are still valid after transformation
                # Convert keypoints back to polygon format and denormalize
                new_polygon = []
                for i in range(0, len(transformed['keypoints']), 2):
                    x = transformed['keypoints'][i] * image_width
                    y = transformed['keypoints'][i + 1] * image_height
                    new_polygon.extend([x, y])
                
                # Only add if the transformed polygon is valid
                if is_valid_polygon(new_polygon):
                    new_ann['segmentation'] = [new_polygon]
                    transformed_annotations.append(new_ann)
    
    return transformed_annotations


def apply_augmentation(
    img: Image.Image,
    annotations: List[Dict[str, Any]],
    parameters: Dict[str, Any]
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Apply Albumentations augmentation pipeline to a PIL Image and its annotations.
    
    Args:
        img: PIL Image to augment
        annotations: List of COCO annotations for the image
        parameters: Parameters for the augmentation pipeline
        
    Returns:
        Tuple of (augmented PIL Image, transformed annotations)
    """
    # Convert PIL to numpy
    img_np = pil_to_numpy(img)
    
    # Get augmentation pipeline
    transform = get_augmentation_pipeline(parameters=parameters)
    
    # Prepare dummy data for the transform
    dummy_data = {
        'image': img_np,
        'bboxes': [],
        'labels': [],
        'keypoints': [],
    }
    
    # Apply augmentation to image only first
    augmented = transform(**dummy_data)
    processed_img = numpy_to_pil(augmented["image"])
    
    # Transform annotations
    transformed_annotations = transform_annotations(
        annotations,
        transform,
        img.height,
        img.width
    )
    
    return processed_img, transformed_annotations


def process_images(
    input_images_dir: str,
    input_coco: Dict[str, Any],
    parameters: Dict[str, Any],
    output_images_dir: str,
    output_coco: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process images and their annotations using Albumentations augmentations.
    """
    os.makedirs(output_images_dir, exist_ok=True)

    # Get all input images
    image_paths = glob(os.path.join(input_images_dir, "*"))

    for image_path in image_paths:
        image_filename = os.path.basename(image_path)

        # Open the image
        img = Image.open(image_path).convert("RGB")

        # Get annotations for this image
        input_image_id = get_image_id_by_filename(input_coco, image_filename)
        annotations = [
            annotation
            for annotation in input_coco["annotations"]
            if annotation["image_id"] == input_image_id
        ]

        # Apply augmentation to image and annotations
        processed_img, transformed_annotations = apply_augmentation(img, annotations, parameters)

        # Save the processed image
        processed_img.save(os.path.join(output_images_dir, image_filename))

        # Register the processed image in COCO metadata
        new_image_id = len(output_coco["images"])
        output_coco["images"].append(
            {
                "id": new_image_id,
                "file_name": image_filename,
                "width": processed_img.width,
                "height": processed_img.height,
            }
        )
        # Add transformed annotations
        for annotation in transformed_annotations:
            new_annotation = deepcopy(annotation)
            new_annotation["image_id"] = new_image_id
            new_annotation["id"] = len(output_coco["annotations"])
            output_coco["annotations"].append(new_annotation)

    print(f"✅ Processed {len(image_paths)} images.")
    return output_coco

def get_image_id_by_filename(coco_data: Dict[str, Any], filename: str) -> int:
    """
    Retrieve the image ID for a given filename.

    Args:
        coco_data (Dict): COCO dataset structure containing images.
        filename (str): Filename of the image.

    Returns:
        int: ID of the image.
    """
    for image in coco_data["images"]:
        if image["file_name"] == filename:
            return image["id"]
    raise ValueError(f"⚠️ Image with filename '{filename}' not found.")


import json
import os
from typing import Union, Dict, List

import cv2
import numpy as np
from imutils import perspective

from src.picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from pipelines.paddle_ocr.pipeline_utils.dataset.paddle_ocr_dataset_context import (
    PaddleOCRDatasetContext,
)


def write_annotations_file(data, output_path):
    """
    Writes the annotation data to a specified file in a text format.

    Args:
        data (List[str]): List of annotation strings to be written to the file.
        output_path (str): The path to the output file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for line in data:
            file.write(line + "\n")


def find_category_id(categories: List[Dict], category_name: str) -> Union[str, None]:
    """
    Finds the ID of a category by its name from a list of categories.

    Args:
        categories (List[Dict]): A list of category dictionaries from the COCO dataset.
        category_name (str): The name of the category to search for.

    Returns:
        Union[str, None]: The ID of the category if found, otherwise None.
    """
    for category in categories:
        if category["name"] == category_name:
            return category["id"]
    return None


def find_category_name(categories: List[Dict], category_id: str) -> Union[str, None]:
    """
    Finds the name of a category by its ID from a list of categories.

    Args:
        categories (List[Dict]): A list of category dictionaries from the COCO dataset.
        category_id (str): The ID of the category to search for.

    Returns:
        Union[str, None]: The name of the category if found, otherwise None.
    """
    for category in categories:
        if category["id"] == category_id:
            return category["name"]
    return None


def find_image_id(images: List[Dict], image_filename: str) -> Union[str, None]:
    """
    Finds the ID of an image by its filename from a list of images.

    Args:
        images (List[Dict]): A list of image dictionaries from the COCO dataset.
        image_filename (str): The filename of the image to search for.

    Returns:
        Union[str, None]: The ID of the image if found, otherwise None.
    """
    for image in images:
        if image["file_name"] == image_filename:
            return image["id"]
    return None


def get_points_from_bbox(bbox: List[int]) -> List[List[int]]:
    """
    Converts a bounding box into a list of points representing its corners.

    Args:
        bbox (List[int]): A list representing the bounding box [x, y, width, height].

    Returns:
        List[List[int]]: A list of points representing the corners of the bounding box.
    """
    x, y, w, h = bbox
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def get_bbox_annotations(coco: Dict, image_directory: str):
    """
    Retrieves and formats bounding box annotations from the COCO data.

    Args:
        coco (Dict): The COCO data containing images, annotations, and categories.
        image_directory (str): The directory containing the images.

    Returns:
        List[str]: A list of formatted bounding box annotations for PaddleOCR.
    """
    processed_data: List[str] = []
    paddle_ocr_annotations: List[Dict] = []
    group_image_id = None

    def append_annotations():
        if group_image_id is not None:
            image_path = os.path.join(
                image_directory, coco["images"][group_image_id]["file_name"]
            )
            processed_data.append(
                f"{image_path}\t{json.dumps(paddle_ocr_annotations, ensure_ascii=False)}"
            )

    for annotation in coco["annotations"]:
        current_image_id = annotation["image_id"]
        if group_image_id is None:
            group_image_id = current_image_id

        if current_image_id != group_image_id:
            append_annotations()
            paddle_ocr_annotations = []
            group_image_id = current_image_id

        paddle_ocr_annotation = {
            "transcription": find_category_name(
                coco["categories"], annotation["category_id"]
            ),
            "points": get_points_from_bbox(annotation["bbox"]),
        }
        paddle_ocr_annotations.append(paddle_ocr_annotation)

    append_annotations()

    return processed_data


def get_text_annotations(coco: Dict, image_directory: str, new_image_directory: str):
    """
    Extracts and processes text annotations from the COCO data by cropping images and saving them.

    Args:
        coco (Dict): The COCO data containing images, annotations, and categories.
        image_directory (str): The directory containing the original images.
        new_image_directory (str): The directory where the cropped images will be saved.

    Returns:
        List[str]: A list of formatted text annotations for PaddleOCR.
    """
    os.makedirs(new_image_directory, exist_ok=True)
    processed_data: List[str] = []
    img_counter = 0

    for annotation in coco["annotations"]:
        current_image_id = annotation["image_id"]
        image_path = os.path.join(
            image_directory, coco["images"][current_image_id]["file_name"]
        )

        points = get_points_from_bbox(annotation["bbox"])

        formatted_points = np.asarray([(x, y) for x, y in points], dtype=np.float32)
        image = cv2.imread(image_path)
        warped = perspective.four_point_transform(image, formatted_points)
        new_image_filename = "img_" + str(img_counter) + ".png"
        new_image_path = os.path.join(new_image_directory, new_image_filename)
        cv2.imwrite(new_image_path, warped)

        processed_data.append(f"{new_image_path}\t{annotation['utf8_string']}")
        img_counter = img_counter + 1

    return processed_data


class PaddleOCRDatasetContextPreparator:
    """
    Prepares and organizes a dataset for OCR tasks using the PaddleOCR format.

    This class takes a dataset context and processes it to extract bounding box and text annotations
    in a format suitable for PaddleOCR.

    Attributes:
        dataset_context (CocoDatasetContext): The context of the dataset to organize.
        destination_path (str): The target directory where the processed dataset will be saved.
        paddle_ocr_dataset_context (PaddleOCRDatasetContext): The prepared dataset context for PaddleOCR.
    """

    def __init__(self, dataset_context: CocoDatasetContext, destination_path: str):
        """
        Initializes the organizer with a given dataset context.

        Args:
            dataset_context (CocoDatasetContext): The dataset context to organize.
            destination_path (str): The directory where the organized dataset will be stored.
        """
        self.dataset_context = dataset_context
        self.destination_path = destination_path
        self.paddle_ocr_dataset_context = PaddleOCRDatasetContext(
            dataset_name=self.dataset_context.dataset_name,
            dataset_version=self.dataset_context.dataset_version,
            assets=self.dataset_context.assets,
            labelmap=self.dataset_context.labelmap,
        )
        self.paddle_ocr_dataset_context.images_dir = self.dataset_context.images_dir
        self.paddle_ocr_dataset_context.annotations_dir = (
            self.dataset_context.annotations_dir
        )
        self.paddle_ocr_dataset_context.coco_file_path = (
            self.dataset_context.coco_file_path
        )
        self.paddle_ocr_dataset_context.coco_data = self.dataset_context.coco_data

    def organize(self) -> PaddleOCRDatasetContext:
        """
        Organizes the dataset context by preparing it for OCR tasks.

        This method processes the COCO data to extract bounding box and text annotations, creates
        the necessary directories, and writes the annotation files in the PaddleOCR format.

        Returns:
            PaddleOCRDatasetContext: The prepared dataset context, ready for OCR tasks.
        """
        if not self.paddle_ocr_dataset_context.coco_data:
            raise ValueError("No COCO data loaded in the dataset context.")
        if not self.paddle_ocr_dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")
        paddleocr_bbox_annotations = get_bbox_annotations(
            coco=self.paddle_ocr_dataset_context.coco_data,
            image_directory=self.paddle_ocr_dataset_context.images_dir,
        )
        self.paddle_ocr_dataset_context.text_images_dir = os.path.join(
            self.destination_path, "text_images"
        )
        paddleocr_text_annotations = get_text_annotations(
            coco=self.paddle_ocr_dataset_context.coco_data,
            image_directory=self.paddle_ocr_dataset_context.images_dir,
            new_image_directory=self.paddle_ocr_dataset_context.text_images_dir,
        )

        self.paddle_ocr_dataset_context.paddle_ocr_bbox_annotations_path = os.path.join(
            self.destination_path,
            "annotations",
            "bbox",
            "annotations.txt",
        )
        self.paddle_ocr_dataset_context.paddle_ocr_text_annotations_path = os.path.join(
            self.destination_path,
            "annotations",
            "text",
            "annotations.txt",
        )
        write_annotations_file(
            data=paddleocr_bbox_annotations,
            output_path=self.paddle_ocr_dataset_context.paddle_ocr_bbox_annotations_path,
        )

        write_annotations_file(
            data=paddleocr_text_annotations,
            output_path=self.paddle_ocr_dataset_context.paddle_ocr_text_annotations_path,
        )
        return self.paddle_ocr_dataset_context

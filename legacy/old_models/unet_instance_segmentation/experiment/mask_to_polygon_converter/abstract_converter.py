import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import approximate_polygon, find_contours
from skimage.transform import resize

from .coco_annotations import COCOAnnotation
from .utils import (
    find_most_similar_string,
    find_similar_string,
    format_polygons,
    shift_x_and_y_coordinates,
)


class AbstractConverter(ABC):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        labelmap: Dict[str, str],
        conversion_tolerance: float = 0.2,
        min_contour_points: int = 10,
    ) -> None:
        """_summary_

        Args:
            images_dir (str): directory of your images with subdirectory as labels
                                example: directory/
                                         ├── car/
                                         │   ├── file1.jpg
                                         │   └── file2.png
                                         ├── bus/
                                         │   ├── file3.jpeg
                                         │   └── file4.jpg

            masks_dir (str): directory of your masks with subdirectory as labels
                                example: directory/
                                         ├── car/
                                         │   ├── file1.jpg
                                         │   └── file2.png
                                         ├── bus/
                                         │   ├── file3.jpeg
                                         │   └── file4.jpg

            labelmap (Dict[str, str]): labelmap of your dataset,
                                example: {'0': 'car', '1': 'bus', '2': 'person'}

            conversion_tolerance (float): The tolerance on the approximation of the polygons extracted from the
            masks. The tolerance must be positive or zero. The smaller the tolerance, the closer the polygon points
            are to each other and therefore the more points there are to describe the polygon.
        """
        self._images_dir = images_dir
        self._masks_dir = masks_dir
        self.labelmap = labelmap
        self.coco_annotations = COCOAnnotation(labelmap=self.labelmap)
        self.conversion_tolerance = conversion_tolerance
        self.min_contour_points = min_contour_points

    def update_coco_annotations(self):
        labels_images_directories = [
            item
            for item in os.listdir(self._images_dir)
            if os.path.isdir(os.path.join(self._images_dir, item))
        ]
        labels_masks_directories = os.listdir(self._masks_dir)

        for label_img_dir in labels_images_directories:
            label_mask_dir = find_most_similar_string(
                label_img_dir, labels_masks_directories
            )
            category_id = self._get_category_id_from_label_directory(
                label_directory=label_img_dir
            )
            img_filenames = os.listdir(os.path.join(self._images_dir, label_img_dir))

            for img_filename in img_filenames:
                img = imread(
                    os.path.join(self._images_dir, label_img_dir, img_filename)
                )
                self.coco_annotations.add_image(
                    img_filename=img_filename,
                    img_height=img.shape[0],
                    img_width=img.shape[1],
                )
                img_id = self.coco_annotations.get_current_img_id()

                masks_filepaths = self._get_masks_filepaths_of_image(
                    img_filename=img_filename, label_dir=label_mask_dir
                )
                for mask_filepath in masks_filepaths:
                    formatted_polygons = self._get_formatted_polygons_from_mask(
                        mask_filepath=mask_filepath, img_shape=img.shape[:1]
                    )
                    for polygon in formatted_polygons:
                        self.coco_annotations.add_polygon_annotation(
                            polygon_list=[polygon],
                            category_id=category_id,
                            img_id=img_id,
                        )
        return self.coco_annotations

    def _get_formatted_polygons_from_mask(
        self, mask_filepath: str, img_shape: Tuple[int, int]
    ) -> List:
        mask = imread(mask_filepath)
        try:
            mask = rgb2gray(mask)
        except ValueError:
            mask = rgb2gray(mask[:, :, :3])
        mask = resize(mask, img_shape)
        polygons = self._convert_mask_to_polygons(mask)
        formatted_polygons = format_polygons(polygons=polygons)
        return formatted_polygons

    def _convert_mask_to_polygons(self, mask: np.ndarray) -> List[np.ndarray]:
        polygons = []
        contours = find_contours(mask)
        for contour in contours:
            approximated_contour = approximate_polygon(
                coords=contour, tolerance=self.conversion_tolerance
            )
            if len(approximated_contour) > self.min_contour_points:
                shifted_contour = shift_x_and_y_coordinates(approximated_contour)
                polygons.append(shifted_contour)
        return polygons

    @abstractmethod
    def _get_masks_filepaths_of_image(
        self, img_filename: str, label_dir: str
    ) -> List[str]:
        pass

    def _get_category_id_from_label_directory(self, label_directory) -> int:
        results = find_similar_string(label_directory, self.labelmap)
        if results is None:
            return -1
        category_key, _ = results
        return int(category_key)

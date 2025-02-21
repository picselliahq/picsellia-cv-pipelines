from abc import ABC
from typing import Dict, List

from .utils import save_dict_as_json_file


class COCOAnnotation(ABC):
    def __init__(self, labelmap: Dict[str, str]):
        self.coco_annotations_dict: Dict[str, List] = {
            "images": [],
            "annotations": [],
            "categories": [],
        }
        self._define_categories(labelmap)

    def get_current_img_id(self):
        return len(self.coco_annotations_dict["images"]) - 1

    def _define_categories(self, labelmap: Dict[str, str]) -> None:
        for category_id, label_name in labelmap.items():
            self._add_category(category_name=label_name, category_id=int(category_id))

    def _add_category(self, category_name: str, category_id: int):
        category = {
            "supercategory": category_name,
            "id": category_id,
            "name": category_name,
        }
        self.coco_annotations_dict["categories"].append(category)

    def add_image(self, img_filename: str, img_height: int, img_width: int):
        img = {
            "file_name": img_filename,
            "height": img_height,
            "width": img_width,
            "id": len(self.coco_annotations_dict["images"]),
        }
        self.coco_annotations_dict["images"].append(img)

    def add_polygon_annotation(
        self,
        polygon_list: List[List[int]],
        category_id: int,
        img_id: int,
    ):
        polygon_annotation = {
            "segmentation": polygon_list,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [],
            "category_id": category_id,
            "id": len(self.coco_annotations_dict["annotations"]),
        }
        self.coco_annotations_dict["annotations"].append(polygon_annotation)

    def save_coco_annotations_as_json(self, json_path):
        save_dict_as_json_file(
            dictionary=self.coco_annotations_dict, json_path=json_path
        )

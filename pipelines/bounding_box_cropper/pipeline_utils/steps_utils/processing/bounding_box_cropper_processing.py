import json
import logging
import os

from picsellia.types.enums import InferenceType
from picsellia_cv_engine.models.data.dataset.dataset_collection import DatasetCollection
from PIL import Image, ImageOps


def open_image_with_exif_rotation(image_filepath: str) -> Image:
    image = Image.open(image_filepath)
    return ImageOps.exif_transpose(image)


class BoundingBoxCropperProcessing:
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.

    It processes the images in the input dataset version to extract the bounding boxes for the specified label.
    The extracted images are saved to the output dataset version.

    Attributes:
        dataset_collection (DatasetCollection): The dataset collection containing the input and output dataset versions.
        label_name_to_extract (str): The name of the label to extract the bounding boxes for.
    """

    def __init__(
        self,
        dataset_collection: DatasetCollection,
        label_name_to_extract: str,
    ):
        self.dataset_collection = dataset_collection
        self.label_name_to_extract = label_name_to_extract

    def _update_output_dataset_version(self):
        """
        Updates the output dataset version with the description and type.
        """
        output_dataset_type = InferenceType.CLASSIFICATION
        input_dataset_context = self.dataset_collection["input"]
        output_dataset_description = (
            f"Dataset extracted from dataset version "
            f"'{input_dataset_context.dataset_version.version}' "
            f"(id: {input_dataset_context.dataset_version.id}) in dataset "
            f"'{input_dataset_context.dataset_version.name}' with label '{self.label_name_to_extract}'."
        )
        self.dataset_collection["output"].dataset_version.update(
            description=output_dataset_description, type=output_dataset_type
        )

    def _process_dataset_collection(self):
        """
        Processes the images in the input dataset version to extract the bounding boxes for the specified label.
        """
        input_json_coco_file = self.dataset_collection["input"].load_coco_file_data()

        category_id_to_name = {
            cat["id"]: cat["name"] for cat in input_json_coco_file["categories"]
        }

        self.output_coco_data = {
            "info": {},
            "licenses": [],
            "categories": [
                {"id": 1, "name": self.label_name_to_extract, "supercategory": "none"}
            ],
            "images": [],
            "annotations": [],
        }

        annotation_id = 0

        self.dataset_collection["output"].coco_file_path = os.path.join(
            self.dataset_collection["output"].annotations_dir, "extracted_images.json"
        )

        for input_annotation in input_json_coco_file["annotations"]:
            if (
                category_id_to_name[input_annotation["category_id"]]
                == self.label_name_to_extract
            ):
                image_info = next(
                    img
                    for img in input_json_coco_file["images"]
                    if img["id"] == input_annotation["image_id"]
                )
                image_filename = image_info["file_name"]
                image_path = os.path.join(
                    self.dataset_collection["input"].images_dir, image_filename
                )

                if os.path.exists(image_path):
                    with open_image_with_exif_rotation(image_path) as image:
                        x, y, width, height = map(int, input_annotation["bbox"])
                        cropped_image = image.crop((x, y, x + width, y + height))

                        image_asset_id, image_extension = os.path.splitext(
                            image_filename
                        )
                        image_data_id = (
                            self.dataset_collection["input"]
                            .dataset_version.list_assets(ids=[image_asset_id])[0]
                            .data_id
                        )
                        output_filename = f"{image_data_id}_{self.label_name_to_extract}_{x}_{y}_{width}_{height}{image_extension}"

                        output_image_path = os.path.join(
                            self.dataset_collection["output"].images_dir,
                            output_filename,
                        )
                        cropped_image.save(output_image_path)

                        self.output_coco_data["images"].append(
                            {
                                "id": annotation_id,
                                "file_name": output_filename,
                                "width": width,
                                "height": height,
                            }
                        )
                        self.output_coco_data["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": annotation_id,
                                "category_id": 1,
                            }
                        )
                        annotation_id += 1
                else:
                    logging.info(f"Image {image_path} not found")

        with open(self.dataset_collection["output"].coco_file_path, "w") as f:
            json.dump(self.output_coco_data, f, indent=4)

        logging.info("Extraction and COCO file generation completed.")

    def process(self) -> DatasetCollection:
        """
        Processes the images in the dataset version to extract the bounding boxes for the specified label and adds them to the output dataset version.
        """
        self._update_output_dataset_version()
        self._process_dataset_collection()
        return self.dataset_collection

from picsellia import Client
from picsellia.sdk.dataset import DatasetVersion
import os

import albumentations as A
import cv2
import numpy as np
from pycocotools.coco import COCO

from picsellia.types.enums import AnnotationFileType

from utils.mask_augmentation import (
    prepare_mask_directories_for_multilabel,
    convert_seperated_multiclass_masks_to_polygons,
    compute_class_to_pixel_dict,
)

api_token = os.environ["api_token"]
organization_id = os.environ["organization_id"]
job_id = os.environ["job_id"]
if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]
client = Client(api_token=api_token, organization_id=organization_id, host=host)
datalake = client.get_datalake()


def create_data_and_add_to_dataset(filepaths, dataset_version):
    multi_data = datalake.upload_data(filepaths, ["augmented"])
    print(f"Adding new Assets to new dataset version: {dataset_version.version}")
    add_to_ds_job = dataset_version.add_data(multi_data)
    add_to_ds_job.wait_for_done()
    print("Finished adding assets")


job = client.get_job_by_id(job_id)

context = job.sync()["dataset_version_processing_job"]
input_dataset_version_id = context["input_dataset_version_id"]
output_dataset_version = context["output_dataset_version_id"]

parameters = context["parameters"]
number_of_augmentations = int(parameters["number_of_augmentations"])

input_dataset_version: DatasetVersion = client.get_dataset_version_by_id(
    input_dataset_version_id
)
input_dataset_version.download("images")

class_to_pixel_mapping = compute_class_to_pixel_dict(input_dataset_version)

transform = A.Compose(
    [
        A.HorizontalFlip(p=float(parameters["horizontal_flip_probability"])),
        A.RandomBrightnessContrast(
            p=float(parameters["random_brightness_contrast_probability"])
        ),
        A.GaussNoise(p=float(parameters["gauss_noise_probability"])),
    ]
)

root_directory = os.getcwd()
annotation_path = input_dataset_version.export_annotation_file(
    AnnotationFileType.COCO, target_path=root_directory
)
coco = COCO(annotation_path)

cat_ids = coco.getCatIds()

augmented_image_dir = "augmented-images"
augmented_mask_dir = "augmented-masks"
os.makedirs(augmented_image_dir, exist_ok=True)
os.makedirs(augmented_mask_dir, exist_ok=True)

augmented_image_paths = []
for img in coco.imgs.values():
    anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    anns_img = np.zeros((img["height"], img["width"]))
    for ann in anns:
        label_name = coco.cats[ann["category_id"]]["name"]
        anns_img = np.maximum(
            anns_img, coco.annToMask(ann) * (class_to_pixel_mapping[label_name])
        )
    image = cv2.imread(os.path.join("images", img["file_name"]))

    for iteration in range(number_of_augmentations):
        transformed = transform(image=image, mask=anns_img)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        transformed_mask_path = os.path.join(
            augmented_mask_dir, f"augmented-{iteration}-{img['file_name']}"
        )
        transformed_image_path = os.path.join(
            augmented_image_dir, f"augmented-{iteration}-{img['file_name']}"
        )
        cv2.imwrite(transformed_mask_path, transformed_mask)
        cv2.imwrite(transformed_image_path, transformed_image)
        augmented_image_paths.append(transformed_image_path)

dataset = client.get_dataset_by_id(input_dataset_version.origin_id)

output_dataset: DatasetVersion = client.get_dataset_version_by_id(
    output_dataset_version
)

create_data_and_add_to_dataset(augmented_image_paths, output_dataset)

original_mask_directory = augmented_mask_dir
data_directory = augmented_image_dir

prepare_mask_directories_for_multilabel(
    class_to_pixel_mapping=class_to_pixel_mapping,
    mask_directory=original_mask_directory,
)

convert_seperated_multiclass_masks_to_polygons(
    data_directory=data_directory, dataset_version=output_dataset
)

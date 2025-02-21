from picsellia import Client
from picsellia.sdk.dataset import DatasetVersion
from picsellia.types.enums import AnnotationFileType
import os
import json
import albumentations as A

from pycocotools.coco import COCO

from ..utils.dict_updates import (
    get_id_from_filename,
    read_img,
    get_annotations,
    save_augmented_image,
    save_image_in_annotations_dict,
    save_annotations_in_annotations_dict,
)

api_token = os.environ["api_token"]
organization_id = os.environ["organization_id"]
job_id = os.environ["job_id"]

if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]

client = Client(api_token=api_token, organization_id=organization_id, host=host)
job = client.get_job_by_id(job_id)

context = job.sync()["dataset_version_processing_job"]
input_dataset_version_id = context["input_dataset_version_id"]
output_dataset_version = context["output_dataset_version_id"]
parameters = context["parameters"]

dataset_path = "data"
augmented_dataset_path = "augmented_data"

if not os.path.exists(augmented_dataset_path):
    os.makedirs(augmented_dataset_path)

input_dataset_version: DatasetVersion = client.get_dataset_version_by_id(
    input_dataset_version_id
)
input_dataset_version.download(dataset_path)

input_assets = input_dataset_version.list_assets()

input_annotations_path = input_dataset_version.export_annotation_file(
    annotation_file_type=AnnotationFileType.COCO,
    target_path=os.path.join("annotations"),
    force_replace=True,
)

input_annotations_coco = COCO(input_annotations_path)

input_annotations_file = open(input_annotations_path)
input_annotations_dict = json.load(input_annotations_file)

output_annotations_dict = input_annotations_dict.copy()
output_annotations_dict["images"] = []
output_annotations_dict["annotations"] = []

augmentation_list = A.Compose(
    [A.Rotate(p=parameters["probability"], limit=parameters["rotation_limit"])],
    bbox_params=A.BboxParams(format="coco"),
    keypoint_params=A.KeypointParams(format="xy"),
)

augmented_img_id = 0
annotation_id = 0

for asset in input_assets:
    image_filename = asset.filename
    w, h = asset.width, asset.height
    img_id = get_id_from_filename(image_filename, input_annotations_dict)

    asset_filepath = os.path.join(dataset_path, image_filename)
    asset_image = read_img(asset_filepath)

    coco_annotations = input_annotations_coco.loadAnns(
        input_annotations_coco.getAnnIds(imgIds=[img_id])
    )
    labels, bboxes, keypoints = get_annotations(coco_annotations, w, h)

    augmented_asset = augmentation_list(
        image=asset_image, labels=labels, bboxes=bboxes, keypoints=keypoints
    )

    augmented_img_filename = (
        image_filename.split(".")[0]
        + "_"
        + format(0)
        + "."
        + image_filename.split(".")[1]
    )
    augmented_image = save_augmented_image(
        augmented_asset["image"], augmented_img_filename, augmented_dataset_path
    )

    output_annotations_dict = save_image_in_annotations_dict(
        augmented_img_filename,
        augmented_img_id,
        augmented_image.width,
        augmented_image.height,
        output_annotations_dict,
    )
    output_annotations_dict = save_annotations_in_annotations_dict(
        augmented_asset, augmented_img_id, annotation_id, output_annotations_dict
    )

    augmented_img_id += 1

filenames = os.listdir(augmented_dataset_path)
filepaths = [os.path.join(augmented_dataset_path, filename) for filename in filenames]

datalake = client.get_datalake()
data_list = datalake.upload_data(filepaths, tags=["augmented", "processing"])

output_dataset: DatasetVersion = client.get_dataset_version_by_id(
    output_dataset_version
)

add_data_job = output_dataset.add_data(data_list)

with open("annotations/augmented_annotations.json", "w") as outfile:
    json.dump(output_annotations_dict, outfile)

add_data_job.wait_for_done()
output_dataset.import_annotations_coco_file(
    file_path="annotations/augmented_annotations.json",
    force_create_label=True,
    fail_on_asset_not_found=False,
)

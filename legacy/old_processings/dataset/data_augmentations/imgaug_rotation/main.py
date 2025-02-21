import os

from random import shuffle
import imgaug.augmenters as iaa
from picsellia import Client
from picsellia.sdk.dataset import DatasetVersion
from picsellia.types.enums import JobStatus

from utils.dict_updates_imgaug import (
    load_image_asset,
    pascal_voc_to_coco,
    format_imgaug_poly_to_int_list,
    save_image,
    save_annotations_in_annotations_dict,
)
from utils.simplify_main import (
    load_input_output_annotations,
    list_filepaths_from_folder,
    upload_data_to_datalake,
    write_annotations_dict,
)
from ..utils.dict_updates import (
    get_id_from_filename,
    get_annotations,
    save_image_in_annotations_dict,
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
input_assets = list(input_dataset_version.list_assets())

(
    input_annotations_dict,
    input_annotations_coco,
    output_annotations_dict,
) = load_input_output_annotations(input_dataset_version)

# sometimes = lambda aug: iaa.Sometimes(parameters['percentage_augmented_imgs'], aug)
# augment = sometimes(iaa.Rotate(rotate=(-parameters['rotation_limit'], parameters['rotation_limit'])))

augment = iaa.Sometimes(
    1, iaa.Rotate(rotate=(-parameters["rotation_limit"], parameters["rotation_limit"]))
)

nb_images_to_augment = int(len(input_assets) * parameters["percentage_augmented_imgs"])
shuffle(input_assets)
assets_to_augment = input_assets[:nb_images_to_augment]
assets_not_to_augment = input_assets[nb_images_to_augment:]

nb_augmented_imgs_per_img = parameters["nb_augmentation_per_img"]
output_img_id = 0
annotation_id = 0

for asset in assets_to_augment:
    for i in range(nb_augmented_imgs_per_img):
        image_filename = asset.filename
        w, h = asset.width, asset.height
        img_id = get_id_from_filename(image_filename, input_annotations_dict)

        asset_image = load_image_asset(dataset_path, image_filename)

        coco_annotations = input_annotations_coco.loadAnns(
            input_annotations_coco.getAnnIds(imgIds=[img_id])
        )
        labels, bboxes, polygons = get_annotations(coco_annotations, w, h)

        augmented_image, augmented_bboxes, augmented_polygons = augment(
            image=asset_image, bounding_boxes=bboxes, polygons=polygons
        )

        if len(augmented_bboxes) > 0:
            corrected_bboxes = list(
                map(lambda box: box.clip_out_of_image((h, w)), augmented_bboxes)
            )
            corrected_bboxes = list(map(pascal_voc_to_coco, corrected_bboxes))
        else:
            corrected_bboxes = []

        if len(augmented_polygons) > 0:
            corrected_polygons = list(
                map(lambda poly: poly.clip_out_of_image((h, w)), augmented_polygons)
            )
            corrected_polygons = list(
                map(format_imgaug_poly_to_int_list, corrected_polygons)
            )
        else:
            corrected_polygons = []

        augmented_img_filename = (
            image_filename.split(".")[0]
            + "_"
            + format(i)
            + "."
            + image_filename.split(".")[1]
        )
        augmented_image = save_image(
            augmented_image, augmented_img_filename, augmented_dataset_path
        )

        output_annotations_dict = save_image_in_annotations_dict(
            augmented_img_filename,
            output_img_id,
            augmented_image.width,
            augmented_image.height,
            output_annotations_dict,
        )

        output_annotations_dict, annotation_id = save_annotations_in_annotations_dict(
            labels,
            corrected_bboxes,
            corrected_polygons,
            output_img_id,
            annotation_id,
            output_annotations_dict,
        )

        output_img_id += 1

for asset in assets_not_to_augment:
    image_filename = asset.filename
    w, h = asset.width, asset.height
    img_id = get_id_from_filename(image_filename, input_annotations_dict)

    asset_image = load_image_asset(dataset_path, image_filename)
    image = save_image(asset_image, image_filename, augmented_dataset_path)
    output_annotations_dict = save_image_in_annotations_dict(
        image_filename,
        output_img_id,
        image.width,
        image.height,
        output_annotations_dict,
    )

    coco_annotations = input_annotations_coco.loadAnns(
        input_annotations_coco.getAnnIds(imgIds=[img_id])
    )

    labels = list(map(lambda annot: annot["category_id"], coco_annotations))
    bboxes = list(map(lambda annot: annot["bbox"], coco_annotations))
    polygons = list(map(lambda annot: annot["segmentation"], coco_annotations))

    output_annotations_dict, annotation_id = save_annotations_in_annotations_dict(
        labels, bboxes, polygons, output_img_id, annotation_id, output_annotations_dict
    )

    output_img_id += 1

output_dataset: DatasetVersion = client.get_dataset_version_by_id(
    output_dataset_version
)

output_dataset.set_type(input_dataset_version.type)

filepaths = list_filepaths_from_folder(augmented_dataset_path)

uploaded_data = upload_data_to_datalake(client, filepaths)

add_data_job = output_dataset.add_data(uploaded_data)

augmented_annotations_path = "annotations/augmented_annotations.json"
write_annotations_dict(output_annotations_dict, augmented_annotations_path)

add_data_job.wait_for_status(
    statuses=JobStatus.SUCCESS, blocking_time_increment=100.0, attempts=100
)
output_dataset.import_annotations_coco_file(
    file_path=augmented_annotations_path,
    force_create_label=True,
    fail_on_asset_not_found=False,
)

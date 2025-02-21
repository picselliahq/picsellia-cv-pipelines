import os
import cv2
import random
from PIL import Image

from picsellia.exceptions import ResourceNotFoundError


def get_id_from_filename(filename, annot_dict):
    for img in annot_dict["images"]:
        if img["file_name"] == filename:
            return img["id"]
    return None


def read_img(filepath):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def correct_bboxes(box, image_width, image_height):
    if box[2] > 0 and box[3] > 0:
        if box[0] < 0:
            box[0] = 0
        if box[1] < 0:
            box[1] = 0
        if box[0] + box[2] > image_width:
            box[2] -= image_width - (box[0] + box[2])
        if box[1] + box[3] > image_height:
            box[3] -= image_height - (box[1] + box[3])
        return box
    else:
        return None


def increment_labels_list(annotation, labels_list):
    labels_list.append(annotation["category_id"])
    return labels_list


def increment_bboxes_list(annotation, image_width, image_height, bboxes_list):
    incremented = True
    if len(annotation["bbox"]) > 0:
        box = [
            annotation["bbox"][0],
            annotation["bbox"][1],
            annotation["bbox"][2],
            annotation["bbox"][3],
        ]
        corrected_box = correct_bboxes(box, image_width, image_height)
        if corrected_box is not None:
            bboxes_list.append(corrected_box)
        else:
            incremented = False
    return bboxes_list, incremented


def increment_polygons_list(annotation, polygons_list):
    if len(annotation["segmentation"]) > 0:
        keypoints = [
            (
                annotation["segmentation"][0][i],
                annotation["segmentation"][0][i + 1],
                i // 2,
            )
            for i in range(0, len(annotation["segmentation"][0]), 2)
        ]
        polygons_list.append(keypoints)
    return polygons_list


def get_annotations(coco_annotations, image_width, image_height):
    labels = []
    bboxes = []
    polygons = []
    for ann in coco_annotations:
        bboxes, incremented = increment_bboxes_list(
            ann, image_width, image_height, bboxes
        )
        if incremented is True:
            labels = increment_labels_list(ann, labels)
        polygons = increment_polygons_list(ann, polygons)
    return labels, bboxes, polygons


def save_augmented_image(
    augmented_image, augmented_asset_filename, augmented_dataset_path
):
    augmented_asset_filepath = os.path.join(
        augmented_dataset_path, augmented_asset_filename
    )
    img = Image.fromarray(augmented_image).convert("RGB")
    img.save(augmented_asset_filepath, "JPEG")
    return img


def save_image_in_annotations_dict(
    asset_filename, img_id, image_width, image_height, annotations_dict
):
    dict_img = {}
    dict_img["file_name"] = asset_filename
    dict_img["width"] = image_width
    dict_img["height"] = image_height
    dict_img["id"] = img_id

    annotations_dict["images"].append(dict_img)

    return annotations_dict


def save_annotations_in_annotations_dict(
    asset, img_id, current_annot_id, annotations_dict
):
    for num_annot in range(len(asset["labels"])):
        dict_annot = {}
        if len(asset["keypoints"]) > 0:
            dict_annot["segmentation"] = asset["keypoints"][num_annot]
        else:
            dict_annot["segmentation"] = []
        dict_annot["iscrowd"] = 0
        dict_annot["image_id"] = img_id
        if len(asset["bboxes"]) > 0:
            dict_annot["bbox"] = list(map(int, asset["bboxes"][num_annot]))
        else:
            dict_annot["bbox"] = []
        dict_annot["category_id"] = int(asset["labels"][num_annot])
        dict_annot["id"] = current_annot_id

        annotations_dict["annotations"].append(dict_annot)
        current_annot_id += 1

    return annotations_dict


def get_or_create_dataset(client, dataset_name, dataset_description):
    try:
        dataset = client.get_dataset(name=dataset_name)
    except ResourceNotFoundError:
        dataset = client.create_dataset(
            name=dataset_name, description=dataset_description, private=True
        )
    return dataset


def get_or_create_dataset_version(
    dataset, dataset_version_name, dataset_version_description, dataset_type
):
    try:
        dataset_version = dataset.get_version(version=dataset_version_name)
    except ResourceNotFoundError:
        dataset_version = dataset.create_version(
            version=dataset_version_name,
            description=dataset_version_description,
            type=dataset_type,
        )
    return dataset_version


def apply_random_augmentation(asset_image, bboxes, labels, augmentation_list):
    num_aug = random.randint(0, len(augmentation_list) - 1)
    aug_type = augmentation_list[num_aug]
    return aug_type(image=asset_image, bboxes=bboxes, labels=labels)

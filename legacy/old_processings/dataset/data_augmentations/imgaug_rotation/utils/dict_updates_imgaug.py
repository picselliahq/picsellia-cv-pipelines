import os
import random

import cv2
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.polys import Polygon
from picsellia.exceptions import ResourceNotFoundError
from PIL import Image


def get_id_from_filename(filename, annot_dict):
    for img in annot_dict["images"]:
        if img["file_name"] == filename:
            return img["id"]
    return None


def read_img(filepath):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_asset(dataset_path, image_filename):
    asset_filepath = os.path.join(dataset_path, image_filename)
    asset_image = read_img(asset_filepath)
    return asset_image


def correct_bboxes(box, image_width, image_height):
    if box[0] < 0:
        box[0] = 0
    if box[1] < 0:
        box[1] = 0
    if box[0] + box[2] > image_width:
        box[2] -= (box[0] + box[2]) - image_width
    if box[1] + box[3] > image_height:
        box[3] -= (box[1] + box[3]) - image_height
    if box[2] > 0 and box[3] > 0:
        return box
    else:
        return None


def increment_labels_list(annotation, labels_list):
    labels_list.append(annotation["category_id"])
    return labels_list


def coco_to_pascal_voc(box):
    x1, y1, w, h = box
    return [int(x1), int(y1), int(x1 + w), int(y1 + h)]


def pascal_voc_to_coco(box):
    return [box.x1_int, box.y1_int, int(box.width), int(box.height)]


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
            box_formatted = coco_to_pascal_voc(corrected_box)
            imgaug_box = BoundingBox(
                x1=box_formatted[0],
                y1=box_formatted[1],
                x2=box_formatted[2],
                y2=box_formatted[3],
                label=annotation["category_id"],
            )
            bboxes_list.append(imgaug_box)
        else:
            incremented = False
    return bboxes_list, incremented


def increment_polygons_list(annotation, polygons_list):
    if len(annotation["segmentation"]) > 0:
        polygon = [
            (annotation["segmentation"][0][i], annotation["segmentation"][0][i + 1])
            for i in range(0, len(annotation["segmentation"][0]), 2)
        ]
        imgaug_polygon = Polygon(polygon, label=annotation["category_id"])
        polygons_list.append(imgaug_polygon)
    return polygons_list


def get_annotations(coco_annotations, image_width, image_height):
    labels = []
    bboxes = []
    polygons = []
    for ann in coco_annotations:
        bboxes, incremented = increment_bboxes_list(
            ann, image_width, image_height, bboxes
        )
        if incremented:
            labels = increment_labels_list(ann, labels)
        polygons = increment_polygons_list(ann, polygons)
    return labels, bboxes, polygons


def save_image(image, asset_filename, dataset_path):
    augmented_asset_filepath = os.path.join(dataset_path, asset_filename)
    img = Image.fromarray(image).convert("RGB")
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


def format_imgaug_poly_to_int_list(polygon):
    xx = polygon[0].xx_int
    yy = polygon[0].yy_int
    formatted_polygon = [None] * (len(xx) + len(yy))
    formatted_polygon[::2] = xx
    formatted_polygon[1::2] = yy
    return [list(map(int, formatted_polygon))]


def save_annotations_in_annotations_dict(
    labels, bboxes, polygons, img_id, current_annot_id, annotations_dict
):
    for num_annot in range(len(labels)):
        dict_annot = {}
        if len(polygons) > 0:
            dict_annot["segmentation"] = polygons[num_annot]
        else:
            dict_annot["segmentation"] = polygons
        dict_annot["iscrowd"] = 0
        dict_annot["image_id"] = img_id
        if len(bboxes) > 0:
            dict_annot["bbox"] = bboxes[num_annot]
        else:
            dict_annot["bbox"] = bboxes
        dict_annot["category_id"] = int(labels[num_annot])
        dict_annot["id"] = current_annot_id

        annotations_dict["annotations"].append(dict_annot)
        current_annot_id += 1

    return annotations_dict, current_annot_id


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

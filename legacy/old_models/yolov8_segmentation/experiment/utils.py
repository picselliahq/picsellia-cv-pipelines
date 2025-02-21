import os

import numpy as np
from picsellia.sdk.experiment import Experiment
from pycocotools.coco import COCO


def create_yolo_segmentation_label(
    experiment: Experiment,
    data_type: str,
    annotations_dict: dict,
    annotations_coco: COCO,
    label_names: list,
):
    dataset_path = os.path.join(experiment.png_dir, data_type)
    image_filenames = os.listdir(os.path.join(dataset_path, "images"))

    labels_path = os.path.join(dataset_path, "labels")

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    for img in annotations_dict["images"]:
        img_filename = img["file_name"]
        if img_filename in image_filenames:
            create_img_label_segmentation(
                img, annotations_coco, labels_path, label_names
            )


def create_img_label_segmentation(
    image: dict, annotations_coco: COCO, labels_path: str, label_names: list
):
    result = []
    img_id = image["id"]
    img_filename = image["file_name"]
    w = image["width"]
    h = image["height"]
    txt_name = os.path.splitext(img_filename)[0] + ".txt"
    annotation_ids = annotations_coco.getAnnIds(imgIds=img_id)
    anns = annotations_coco.loadAnns(annotation_ids)
    for ann in anns:
        seg = coco_to_yolo_segmentation(ann["segmentation"], w, h)
        seg_string = " ".join([str(x) for x in seg])
        label = label_names.index(
            annotations_coco.loadCats(ann["category_id"])[0]["name"]
        )
        result.append(f"{label} {seg_string}")
    with open(os.path.join(labels_path, txt_name), "w") as f:
        f.write("\n".join(result))


def coco_to_yolo_segmentation(ann: list, image_w: int, image_h: int) -> list:
    pair_index = np.arange(0, len(ann[0]), 2)
    impair_index = np.arange(1, len(ann[0]), 2)
    Xs = list(map(ann[0].__getitem__, pair_index))
    xs = list(map(lambda x: x / image_w, Xs))
    Ys = list(map(ann[0].__getitem__, impair_index))
    ys = list(map(lambda x: x / image_h, Ys))
    return interleave_lists(xs, ys)


def interleave_lists(lst1: list, lst2: list) -> list:
    return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]

import json
import logging
import math
import os
from typing import Union

import numpy as np
import picsellia
import requests
import torch
import tqdm
from evaluator.type_formatter import TypeFormatter
from evaluator.utils.general import transpose_if_exif_tags
from picsellia import Asset, Job
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import InferenceType
from PIL import Image, UnidentifiedImageError
from pycocotools.coco import COCO
from YOLOX.tools.demo import Predictor


class YOLOV8StyleOutput:
    class Boxes:
        def __init__(self, boxes, conf, cls):
            self.xyxyn = boxes
            self.conf = conf
            self.cls = cls

    def __init__(self, yolox_output: torch.Tensor, img_info: dict):
        self.img_width = img_info["width"]
        self.img_height = img_info["height"]
        self.ratio = img_info["ratio"]

        # Step 1: Unscale to original image size
        boxes = yolox_output[:, 0:4].clone()  # (x1, y1, x2, y2)

        boxes /= self.ratio  # reproject to original image size

        # Step 2: Normalize coordinates between 0 and 1
        normalized_boxes = torch.zeros_like(boxes)
        normalized_boxes[:, 0] = boxes[:, 0] / self.img_width
        normalized_boxes[:, 1] = boxes[:, 1] / self.img_height
        normalized_boxes[:, 2] = boxes[:, 2] / self.img_width
        normalized_boxes[:, 3] = boxes[:, 3] / self.img_height

        conf = yolox_output[:, 4] * yolox_output[:, 5]
        cls = yolox_output[:, 6]

        self.boxes = self.Boxes(normalized_boxes, conf, cls)

    @property
    def probs(self):
        return self.boxes.conf


def create_yolo_detection_label(
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
            create_img_label_detection(img, annotations_coco, labels_path, label_names)


def create_img_label_detection(
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
        bbox = ann["bbox"]
        yolo_bbox = coco_to_yolo_detection(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        seg_string = " ".join([str(x) for x in yolo_bbox])
        label = label_names.index(
            annotations_coco.loadCats(ann["category_id"])[0]["name"]
        )
        result.append(f"{label} {seg_string}")
    with open(os.path.join(labels_path, txt_name), "w") as f:
        f.write("\n".join(result))


def coco_to_yolo_detection(
    x1: int, y1: int, w: int, h: int, image_w: int, image_h: int
) -> list[float]:
    return [
        ((2 * x1 + w) / (2 * image_w)),
        ((2 * y1 + h) / (2 * image_h)),
        w / image_w,
        h / image_h,
    ]


def evaluate_model(
    yolox_predictor: Predictor,
    type_formatter: TypeFormatter,
    experiment: Experiment,
    asset_list: MultiAsset,
    dataset_type: InferenceType,
    confidence_threshold: float = 0.1,
) -> Job:
    evaluation_batch_size = experiment.get_log(name="parameters").data.get(
        "evaluation_batch_size", 8
    )
    batch_size = (
        evaluation_batch_size
        if len(asset_list) > evaluation_batch_size
        else len(asset_list)
    )
    total_batch_number = math.ceil(len(asset_list) / batch_size)

    for i in tqdm.tqdm(range(total_batch_number)):
        subset_asset_list = asset_list[i * batch_size : (i + 1) * batch_size]
        inputs = preprocess_images(subset_asset_list)

        for j, asset in enumerate(subset_asset_list):
            prediction, img_info = yolox_predictor.inference(inputs[j])

            evaluations: dict[str, list] = {"rectangles": []}

            if prediction[0] is not None:
                yolov8_style_output = YOLOV8StyleOutput(
                    yolox_output=prediction[0], img_info=img_info
                )

                evaluations["rectangles"] = format_prediction_to_evaluations(
                    asset=asset,
                    prediction=yolov8_style_output,
                    type_formatter=type_formatter,
                    confidence_threshold=confidence_threshold,
                )

            send_evaluations_to_platform(
                experiment=experiment, asset=asset, evaluations=evaluations
            )

    if dataset_type in [
        InferenceType.OBJECT_DETECTION,
        InferenceType.SEGMENTATION,
        InferenceType.CLASSIFICATION,
    ]:
        return experiment.compute_evaluations_metrics(inference_type=dataset_type)


def preprocess_images(assets: list[Asset]) -> list[np.array]:
    images = []
    for asset in assets:
        try:
            image = open_asset_as_array(asset)
        except UnidentifiedImageError:
            logging.warning(f"Can't evaluate {asset.filename}, error opening the image")
            continue
        images.append(image)
    return images


def open_asset_as_array(asset: Asset) -> np.array:
    image = Image.open(requests.get(asset.reset_url(), stream=True).raw)
    image = transpose_if_exif_tags(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def send_evaluations_to_platform(
    experiment: Experiment, asset: Asset, evaluations: dict
) -> None:
    try:
        experiment.add_evaluation(asset=asset, **evaluations)
        print(f"Asset: {asset.filename} evaluated.")
        logging.info(f"Asset: {asset.filename} evaluated.")

    except Exception:
        logging.info(
            f"Something went wrong with evaluating {asset.filename}. Skipping.."
        )


def format_prediction_to_evaluations(
    asset: Asset,
    prediction: Union[list, dict, YOLOV8StyleOutput],
    type_formatter: TypeFormatter,
    confidence_threshold: float,
) -> list:
    picsellia_predictions = type_formatter.format_prediction(
        asset=asset, prediction=prediction
    )

    evaluations = []
    for i in range(min(100, len(picsellia_predictions["confidences"]))):
        if picsellia_predictions["confidences"][i] >= confidence_threshold:
            picsellia_prediction = {
                prediction_key: prediction[i]
                for prediction_key, prediction in picsellia_predictions.items()
            }
            evaluation = type_formatter.format_evaluation(
                picsellia_prediction=picsellia_prediction
            )
            evaluations.append(evaluation)
    return evaluations


def extract_dataset_assets(experiment: Experiment, prop_train_split: float) -> tuple:
    attached_datasets = experiment.list_attached_dataset_versions()
    base_imgdir = experiment.png_dir

    if len(attached_datasets) == 3:
        return _handle_three_attached_datasets(
            experiment=experiment,
            base_imgdir=base_imgdir,
        )

    elif len(attached_datasets) == 2:
        return _handle_two_attached_datasets(
            experiment=experiment,
            base_imgdir=base_imgdir,
            prop_train_split=prop_train_split,
        )

    elif len(attached_datasets) == 1:
        return _handle_one_attached_dataset(
            experiment=experiment,
            base_imgdir=base_imgdir,
            prop_train_split=prop_train_split,
        )
    else:
        raise picsellia.exceptions.PicselliaError(
            "This model expect 3 datasets: `train`, `test` and `val`."
        )


def _handle_three_attached_datasets(experiment: Experiment, base_imgdir: str) -> tuple:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception as e:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'train' dataset. Expecting 'train', 'test', 'val'"
        ) from e
    try:
        test_ds = experiment.get_dataset(name="test")
    except Exception as e:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'test' dataset. Expecting 'train', 'test', 'val'"
        ) from e
    try:
        val_ds = experiment.get_dataset(name="val")
    except Exception as e:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'val' dataset. Expecting 'train', 'test', 'val"
        ) from e

    label_names = [label.name for label in train_ds.list_labels()]

    for data_type, dataset in {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    }.items():
        asset_list = dataset.list_assets()
        coco_annotation = dataset.build_coco_file_locally(
            assets=asset_list, enforced_ordered_categories=label_names, use_id=True
        )
        annotations_dict = coco_annotation.dict()
        annotations_path = os.path.join(base_imgdir, f"{data_type}_annotations.json")

        with open(annotations_path, "w") as f:
            f.write(json.dumps(annotations_dict))

        dataset_path = os.path.join(base_imgdir, data_type, "images")
        os.makedirs(dataset_path)

        asset_list.download(target_path=dataset_path, max_workers=8, use_id=True)

    return (
        train_ds.list_assets(),
        test_ds.list_assets(),
        val_ds.list_assets(),
        train_ds.list_labels(),
        test_ds.list_labels(),
        train_ds.type,
    )


def _handle_two_attached_datasets(
    experiment: Experiment, base_imgdir: str, prop_train_split: float
) -> tuple:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception as e:
        raise ResourceNotFoundError(
            "Found 2 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')"
        ) from e
    try:
        test_ds = experiment.get_dataset(name="test")
    except Exception as e:
        raise ResourceNotFoundError(
            "Found  attached datasets, but can't find any 'test' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')"
        ) from e

    assets, classes_repartition, labels = train_ds.split_into_multi_assets(
        ratios=[prop_train_split, 1 - prop_train_split]
    )
    labelmap = {str(i): label.name for i, label in enumerate(labels)}
    label_names = [label.name for label in labels]

    experiment.log(
        "train-split",
        order_repartition_according_labelmap(labelmap, classes_repartition[0]),
        "bar",
        replace=True,
    )
    experiment.log(
        "val-split",
        order_repartition_according_labelmap(labelmap, classes_repartition[1]),
        "bar",
        replace=True,
    )

    for data_type, dataset in {
        "test": test_ds,
    }.items():
        asset_list = dataset.list_assets()
        coco_annotation = dataset.build_coco_file_locally(
            assets=asset_list, enforced_ordered_categories=label_names, use_id=True
        )
        annotations_dict = coco_annotation.dict()
        annotations_path = os.path.join(base_imgdir, f"{data_type}_annotations.json")

        with open(annotations_path, "w") as f:
            f.write(json.dumps(annotations_dict))

        dataset_path = os.path.join(base_imgdir, data_type, "images")
        os.makedirs(dataset_path)

        asset_list.download(target_path=dataset_path, max_workers=8, use_id=True)

    for split, asset_list in {"train": assets[0], "val": assets[1]}.items():
        coco_annotation = train_ds.build_coco_file_locally(
            assets=asset_list, enforced_ordered_categories=label_names, use_id=True
        )
        annotations_dict = coco_annotation.dict()
        annotations_path = os.path.join(base_imgdir, f"{split}_annotations.json")

        with open(annotations_path, "w") as f:
            f.write(json.dumps(annotations_dict))

        dataset_path = os.path.join(base_imgdir, split, "images")
        os.makedirs(dataset_path)

        asset_list.download(target_path=dataset_path, max_workers=8, use_id=True)

    return (
        assets[0],
        test_ds.list_assets(),
        assets[1],
        labels,
        test_ds.list_labels(),
        train_ds.type,
    )


def _handle_one_attached_dataset(
    experiment: Experiment, base_imgdir: str, prop_train_split: float
) -> tuple:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception as e:
        raise ResourceNotFoundError(
            "Found 2 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')"
        ) from e

    assets, classes_repartition, labels = train_ds.split_into_multi_assets(
        ratios=[
            prop_train_split,
            (1.0 - prop_train_split) / 2,
            (1.0 - prop_train_split) / 2,
        ]
    )
    labelmap = {str(i): label.name for i, label in enumerate(labels)}
    label_names = [label.name for label in labels]

    experiment.log(
        "train-split",
        order_repartition_according_labelmap(labelmap, classes_repartition[0]),
        "bar",
        replace=True,
    )
    experiment.log(
        "test-split",
        order_repartition_according_labelmap(labelmap, classes_repartition[1]),
        "bar",
        replace=True,
    )
    experiment.log(
        "val-split",
        order_repartition_according_labelmap(labelmap, classes_repartition[2]),
        "bar",
        replace=True,
    )

    for split, asset_list in {
        "train": assets[0],
        "test": assets[1],
        "val": assets[2],
    }.items():
        coco_annotation = train_ds.build_coco_file_locally(
            assets=asset_list, enforced_ordered_categories=label_names, use_id=True
        )
        annotations_dict = coco_annotation.dict()
        annotations_path = os.path.join(base_imgdir, f"{split}_annotations.json")

        with open(annotations_path, "w") as f:
            f.write(json.dumps(annotations_dict))

        dataset_path = os.path.join(base_imgdir, split, "images")
        os.makedirs(dataset_path)

        asset_list.download(target_path=dataset_path, max_workers=8, use_id=True)

    return (
        assets[0],
        assets[1],
        assets[2],
        labels,
        labels,
        train_ds.type,
    )


def order_repartition_according_labelmap(labelmap: dict, repartition: dict) -> dict:
    ordered_rep = {"x": list(labelmap.values()), "y": []}
    for name in labelmap.values():
        if name in repartition:
            ordered_rep["y"].append(repartition[name])
        else:
            ordered_rep["y"].append(0)
    return ordered_rep

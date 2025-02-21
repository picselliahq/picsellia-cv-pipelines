import logging
import os
import shutil

import numpy
import numpy as np
from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.label import Label
from picsellia.types.enums import AnnotationFileType
from PIL import Image
from pycocotools.coco import COCO
from ultralytics.yolo.engine.model import YOLO


def create_and_log_labelmap(experiment: Experiment) -> dict:
    class_names_list = os.listdir("data/train")
    labelmap = {str(i): label for i, label in enumerate(sorted(class_names_list))}
    experiment.log("labelmap", labelmap, "labelmap", replace=True)
    return labelmap


def prepare_datasets_with_annotation(
    train_set: DatasetVersion, test_set: DatasetVersion, val_set: DatasetVersion
) -> tuple[DatasetVersion, MultiAsset]:
    coco_train, coco_test, coco_val = _create_coco_objects(train_set, test_set, val_set)

    move_files_in_class_directories(coco_train, "data/train")
    move_files_in_class_directories(coco_test, "data/test")
    move_files_in_class_directories(coco_val, "data/val")

    evaluation_ds = test_set
    evaluation_assets = evaluation_ds.list_assets()

    return evaluation_ds, evaluation_assets


def _create_coco_objects(
    train_set: DatasetVersion, test_set: DatasetVersion, val_set: DatasetVersion
) -> tuple[COCO, COCO, COCO]:
    train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)

    test_annotation_path = test_set.export_annotation_file(AnnotationFileType.COCO)
    coco_test = COCO(test_annotation_path)

    val_annotation_path = val_set.export_annotation_file(AnnotationFileType.COCO)
    coco_val = COCO(val_annotation_path)

    return coco_train, coco_test, coco_val


def move_all_files_in_class_directories(train_set: DatasetVersion) -> None:
    train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)
    move_files_in_class_directories(coco_train, "data/train")
    move_files_in_class_directories(coco_train, "data/test")
    move_files_in_class_directories(coco_train, "data/val")


def move_files_in_class_directories(coco: COCO, base_imdir: str = None) -> None | str:
    fnames = os.listdir(base_imdir)
    _create_class_directories(coco=coco, base_imdir=base_imdir)
    for i in coco.imgs:
        image = coco.imgs[i]
        cat = get_image_annotation(coco=coco, fnames=fnames, image=image)
        if cat is None:
            continue
        move_image(
            filename=image["file_name"],
            old_location_path=base_imdir,
            new_location_path=os.path.join(base_imdir, cat["name"]),
        )
    logging.info(f"Formatting {base_imdir} .. OK")
    return base_imdir


def _create_class_directories(coco: COCO, base_imdir: str = None) -> None:
    for i in coco.cats:
        cat = coco.cats[i]
        class_folder = os.path.join(base_imdir, cat["name"])
        if not os.path.isdir(class_folder):
            os.makedirs(class_folder)
    logging.info(f"Formatting {base_imdir} ..")


def get_image_annotation(coco: COCO, fnames: list[str], image: dict) -> None | dict:
    if image["file_name"] not in fnames:
        return None
    ann = coco.loadAnns(coco.getAnnIds(image["id"]))
    if len(ann) > 1:
        logging.info(f"{image['file_name']} has more than one class. Skipping")
    ann = ann[0]
    cat = coco.loadCats(ann["category_id"])[0]

    return cat


def split_single_dataset(
    experiment: Experiment, train_set: DatasetVersion
) -> tuple[
    MultiAsset,
    MultiAsset,
    MultiAsset,
    dict[str, list],
    dict[str, list],
    dict[str, list],
    list[Label],
]:
    parameters = experiment.get_log("parameters").data
    prop = get_prop_parameter(parameters)
    (
        train_assets,
        test_assets,
        eval_assets,
        train_rep,
        test_rep,
        val_rep,
        labels,
    ) = train_set.train_test_val_split([prop, (1 - prop) / 2, (1 - prop) / 2])

    make_train_test_val_dirs()
    move_images_in_train_test_val_folders(
        train_assets=train_assets, test_assets=test_assets, eval_assets=eval_assets
    )

    return train_assets, test_assets, eval_assets, train_rep, test_rep, val_rep, labels


def get_prop_parameter(parameters: dict) -> float:
    prop = parameters.get("prop_train_split", 0.7)
    return prop


def make_train_test_val_dirs() -> None:
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)


def move_images_in_train_test_val_folders(
    train_assets: MultiAsset, test_assets: MultiAsset, eval_assets: MultiAsset
) -> None:
    for asset in train_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/train",
        )
    for asset in test_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/test",
        )

    for asset in eval_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/val",
        )


def move_images_in_train_val_folders(train_assets: MultiAsset, eval_assets: MultiAsset):
    for asset in train_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/train",
        )
    for asset in eval_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path="data/val",
        )


def move_image(filename: str, old_location_path: str, new_location_path: str) -> None:
    old_path = os.path.join(old_location_path, filename)
    new_path = os.path.join(new_location_path, filename)
    try:
        shutil.move(old_path, new_path)
    except Exception:
        logging.info(f"{filename} skipped.")


def download_triple_dataset(
    train_set: DatasetVersion, test_set: DatasetVersion, eval_set: DatasetVersion
) -> None:
    for data_type, dataset in {
        "train": train_set,
        "test": test_set,
        "val": eval_set,
    }.items():
        dataset.download(target_path=os.path.join("data", data_type), max_workers=8)


def log_split_dataset_repartition_to_experiment(
    experiment: Experiment,
    labelmap: dict,
    train_rep: dict[str, list],
    test_rep: dict[str, list],
    val_rep: dict[str, list],
) -> None:
    experiment.log(
        "train-split",
        order_repartition_according_labelmap(labelmap, train_rep),
        "bar",
        replace=True,
    )
    experiment.log(
        "test-split",
        order_repartition_according_labelmap(labelmap, test_rep),
        "bar",
        replace=True,
    )
    experiment.log(
        "val-split",
        order_repartition_according_labelmap(labelmap, val_rep),
        "bar",
        replace=True,
    )


def order_repartition_according_labelmap(labelmap: dict, repartition: dict) -> dict:
    ordered_rep = {"x": list(labelmap.values()), "y": []}
    for name in ordered_rep["x"]:
        ordered_rep["y"].append(repartition["y"][repartition["x"].index(name)])
    return ordered_rep


def predict_evaluation_images(
    labelmap: dict, val_folder_path: str, model: YOLO
) -> tuple[list, list]:
    ground_truths = []
    predictions = []
    for class_id, label in labelmap.items():
        label_path = os.path.join(val_folder_path, label)
        if os.path.exists(label_path):
            file_list = [
                os.path.join(label_path, filepath)
                for filepath in os.listdir(label_path)
            ]
            for image in file_list:
                image = Image.open(image).convert("RGB")
                pred = model(np.array(image))
                pred_label = np.argmax([float(score) for score in list(pred[0].probs)])
                ground_truths.append(int(class_id))
                predictions.append(pred_label)
    return ground_truths, predictions


def log_confusion_to_experiment(
    experiment: Experiment, labelmap: dict, matrix: numpy.ndarray
) -> None:
    confusion = format_confusion_matrix(labelmap=labelmap, matrix=matrix)
    experiment.log(name="confusion", data=confusion, type="heatmap")


def format_confusion_matrix(labelmap: dict, matrix: numpy.ndarray) -> dict:
    return {"categories": list(labelmap.values()), "values": matrix.tolist()}

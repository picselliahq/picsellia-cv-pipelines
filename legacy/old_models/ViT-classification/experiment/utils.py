import logging
import os
import shutil
from typing import Optional

import evaluate
import numpy as np
from picsellia import Experiment
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.dataset_version import DatasetVersion
from picsellia.sdk.label import Label
from picsellia.types.enums import AnnotationFileType
from pycocotools.coco import COCO


def compute_metrics(eval_pred) -> float:
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)


def prepare_datasets_with_annotation(
    train_set: DatasetVersion,
    test_set: DatasetVersion,
    val_set: DatasetVersion,
    train_test_eval_path_dict: dict,
) -> tuple[DatasetVersion, MultiAsset]:
    coco_train, coco_test, coco_val = _create_coco_objects(train_set, test_set, val_set)

    _move_files_in_class_directories(
        coco_train, train_test_eval_path_dict["train_path"]
    )
    _move_files_in_class_directories(coco_test, train_test_eval_path_dict["test_path"])
    _move_files_in_class_directories(coco_val, train_test_eval_path_dict["eval_path"])

    evaluation_ds = val_set
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


def _move_all_files_in_class_directories(
    train_set: DatasetVersion, train_test_eval_path_dict: dict
) -> None:
    train_annotation_path = train_set.export_annotation_file(AnnotationFileType.COCO)
    coco_train = COCO(train_annotation_path)
    _move_files_in_class_directories(
        coco_train, train_test_eval_path_dict["train_path"]
    )
    _move_files_in_class_directories(coco_train, train_test_eval_path_dict["test_path"])
    _move_files_in_class_directories(coco_train, train_test_eval_path_dict["eval_path"])


def _move_files_in_class_directories(
    coco: COCO, base_imdir: Optional[str] = None
) -> None | str:
    if not base_imdir:
        return None
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


def _create_class_directories(coco: COCO, base_imdir: str) -> None:
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


def get_train_test_eval_datasets_from_experiment(
    experiment: Experiment,
) -> tuple[bool, bool, DatasetVersion, DatasetVersion, DatasetVersion]:
    number_of_attached_datasets = len(experiment.list_attached_dataset_versions())
    has_three_datasets, has_one_dataset = False, False
    if number_of_attached_datasets == 3:
        has_three_datasets = True
        train_set, test_set, eval_set = _get_three_attached_datasets(experiment)
    elif number_of_attached_datasets == 1:
        has_one_dataset = True
        logging.info(
            "We found only one dataset inside your experiment, the train/test/split will be performed automatically."
        )
        train_set = experiment.list_attached_dataset_versions()[0]
        test_set = None
        eval_set = None

    else:
        logging.info("We need exactly 1 or 3 datasets attached to this experiment ")

    return has_one_dataset, has_three_datasets, train_set, test_set, eval_set


def _get_three_attached_datasets(
    experiment: Experiment,
) -> tuple[DatasetVersion, DatasetVersion, DatasetVersion]:
    try:
        train_set = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                                expecting 'train', 'test', 'eval')"
        )
    try:
        test_set = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                                expecting 'train', 'test', 'eval')"
        )
    try:
        eval_set = experiment.get_dataset(name="eval")
    except Exception:
        raise ResourceNotFoundError(
            "Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                    expecting 'train', 'test', 'eval')"
        )
    return train_set, test_set, eval_set


def split_single_dataset(
    parameters: dict, train_set: DatasetVersion, train_test_eval_path_dict: dict
) -> tuple[
    MultiAsset,
    MultiAsset,
    MultiAsset,
    dict[str, list],
    dict[str, list],
    dict[str, list],
    list[Label],
]:
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

    make_train_test_val_dirs(train_test_eval_path_dict)
    move_images_in_train_test_val_folders(
        train_assets=train_assets,
        test_assets=test_assets,
        eval_assets=eval_assets,
        train_test_val_path=train_test_eval_path_dict,
    )

    return train_assets, test_assets, eval_assets, train_rep, test_rep, val_rep, labels


def get_prop_parameter(parameters: dict) -> float:
    prop = parameters.get("prop_train_split", 0.7)
    return prop


def make_train_test_val_dirs(train_test_eval_path_dict: dict) -> None:
    os.makedirs(train_test_eval_path_dict["train_path"], exist_ok=True)
    os.makedirs(train_test_eval_path_dict["test_path"], exist_ok=True)
    os.makedirs(train_test_eval_path_dict["eval_path"], exist_ok=True)


def move_images_in_train_test_val_folders(
    train_assets: MultiAsset,
    test_assets: MultiAsset,
    eval_assets: MultiAsset,
    train_test_val_path: dict,
) -> None:
    for asset in train_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path=train_test_val_path["train_path"],
        )
    for asset in test_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path=train_test_val_path["test_path"],
        )

    for asset in eval_assets:
        move_image(
            filename=asset.filename,
            old_location_path="images",
            new_location_path=train_test_val_path["eval_path"],
        )


def move_image(filename: str, old_location_path: str, new_location_path: str) -> None:
    old_path = os.path.join(old_location_path, filename)
    new_path = os.path.join(new_location_path, filename)
    try:
        shutil.move(old_path, new_path)
    except Exception:
        logging.info(f"{filename} skipped.")


def get_predicted_label_confidence(predictions: list) -> tuple[str, float]:
    scores = []
    classes = []
    for pred in predictions:
        scores.append(pred["score"])
        classes.append(pred["label"])

    max_conf = max(scores)

    predicted_class = classes[scores.index(max_conf)]

    return predicted_class, max_conf


def get_asset_filename_from_path(file_path: str) -> str:
    return file_path.split("/")[-1]


def find_asset_by_filename(filename: str, dataset: DatasetVersion):
    try:
        asset = dataset.find_asset(filename=filename)
        return asset
    except Exception as e:
        print(e)
        return None


def log_labelmap(id2label: dict, experiment: Experiment):
    labelmap = {str(k): v for k, v in id2label.items()}
    experiment.log("labelmap", labelmap, "labelmap", replace=True)

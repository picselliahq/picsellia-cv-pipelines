import json
import os

import albumentations
import evaluate
import numpy as np
import torch
import torchvision
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import AnnotationFileType
from tqdm import tqdm
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

transform = albumentations.Compose(
    [
        albumentations.Resize(960, 960),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super().__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        # remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}


def get_id2label_mapping(annotations: dict) -> dict:
    categories = get_category_names(annotations=annotations)
    id2label = dict(enumerate(categories, start=0))

    return id2label


def get_category_names(annotations: dict) -> list[str]:
    return [cat["name"] for cat in annotations["categories"]]


def log_labelmap(id2label: dict, experiment: Experiment):
    labelmap = {str(k): v for k, v in id2label.items()}
    experiment.log("labelmap", labelmap, "labelmap", replace=True)


def create_objects_dict(annotations: dict, image_id: int) -> dict:
    curr_object: dict[str, list] = {
        key: [] for key in ["id", "bbox", "category", "area", "image_id"]
    }

    for ann in annotations["annotations"]:
        if ann["image_id"] == image_id:
            curr_object["id"].append(ann["id"])
            curr_object["bbox"].append(ann["bbox"])
            curr_object["category"].append(ann["category_id"])
            curr_object["area"].append(ann["area"])
            curr_object["image_id"].append(image_id)

    return curr_object


def format_and_write_annotations(dataset: DatasetVersion, data_dir: str) -> dict:
    annotation_file_path = internal_export_annotation_file(
        dataset=dataset, target_path=data_dir
    )
    annotations = read_annotation_file(annotation_file_path)
    formatted_coco = format_coco_annot_to_jsonlines_format(annotations=annotations)
    write_metadata_file(
        data=formatted_coco, output_path=os.path.join(data_dir, "metadata.jsonl")
    )
    return annotations


def internal_export_annotation_file(dataset: DatasetVersion, target_path: str) -> str:
    annotation_file_path = dataset.export_annotation_file(
        annotation_file_type=AnnotationFileType.COCO, target_path=target_path
    )
    return annotation_file_path


def read_annotation_file(annotation_file_path: str) -> dict:
    with open(annotation_file_path) as f:
        annotations = json.load(f)
    return annotations


def export_and_read_annotation_file(dataset: DatasetVersion, target_path: str):
    annotation_path = internal_export_annotation_file(dataset, target_path)
    annotations = read_annotation_file(annotation_file_path=annotation_path)
    return annotations


def format_coco_annot_to_jsonlines_format(annotations: dict) -> list[dict]:
    formatted_coco = []
    for image in annotations["images"]:
        image_id = image["id"]
        one_line = {
            "file_name": image["file_name"],
            "image_id": image_id,
            "width": image["width"],
            "height": image["height"],
            "objects": create_objects_dict(annotations, image_id),
        }
        formatted_coco.append(one_line)

    return formatted_coco


def write_metadata_file(data: list[dict], output_path: str):
    try:
        with open(output_path, "w") as f:
            for entry in data:
                json.dump(entry, f)
                f.write("\n")
    except OSError:
        print("could not write metadata file")


def custom_train_test_eval_split(
    loaded_dataset: DatasetDict, test_prop: float
) -> DatasetDict:
    first_split = loaded_dataset["train"].train_test_split(test_size=test_prop, seed=11)
    test_valid = first_split["test"].train_test_split(test_size=0.5, seed=11)
    train_test_valid_dataset = DatasetDict(
        {
            "train": first_split["train"],
            "test": test_valid["test"],
            "eval": test_valid["train"],
        }
    )
    return train_test_valid_dataset


def transform_images_and_annotations(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(
            image=image, bboxes=objects["bbox"], category=objects["category"]
        )

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_annotations(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


def save_annotation_file_images(
    dataset: Dataset, experiment: Experiment, id2label: dict
) -> tuple[str, str]:
    # Save images and annotations into the test_files torchvision.datasets.CocoDetection expects
    output_json: dict[str, list] = {}
    path_output = os.path.join(experiment.base_dir, "output")

    if not os.path.exists(path_output):
        os.makedirs(path_output)

    path_anno = os.path.join(path_output, "ann.json")
    categories_json = [
        {"supercategory": "none", "id": id, "name": id2label[id]} for id in id2label
    ]
    output_json["images"] = []
    output_json["annotations"] = []
    for example in dataset:
        ann = val_formatted_annotations(example["image_id"], example["objects"])
        output_json["images"].append(
            {
                "id": example["image_id"],
                "width": example["image"].width,
                "height": example["image"].height,
                "file_name": f"{example['image_id']}.png",
            }
        )
        output_json["annotations"].extend(ann)
    output_json["categories"] = categories_json

    with open(path_anno, "w") as file:
        json.dump(output_json, file, ensure_ascii=False, indent=4)

    for im, img_id in zip(dataset["image"], dataset["image_id"]):
        path_img = os.path.join(path_output, f"{img_id}.png")
        im.save(path_img)

    return path_output, path_anno


def formatted_annotations(image_id, category, area, bbox):
    annotations = []

    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels,
    }
    return batch


def val_formatted_annotations(image_id, objects):
    annotations = []
    for i in range(0, len(objects["id"])):
        new_ann = {
            "id": objects["id"][i],
            "category_id": objects["category"][i],
            "iscrowd": 0,
            "image_id": image_id,
            "area": objects["area"][i],
            "bbox": objects["bbox"][i],
        }
        annotations.append(new_ann)

    return annotations


def format_evaluation_results(results: dict) -> dict:
    casted_results = {}
    for metric, value in results["iou_bbox"].items():
        casted_results[metric] = float(value)

    return casted_results


def run_evaluation(test_ds_coco_format, im_processor, model) -> dict:
    module = evaluate.load("ybelkada/cocoevaluate", coco=test_ds_coco_format.coco)
    val_dataloader = torch.utils.data.DataLoader(
        test_ds_coco_format,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        for _idx, batch in enumerate(tqdm(val_dataloader)):
            pixel_values = batch["pixel_values"]
            pixel_mask = batch["pixel_mask"]

            labels = [dict(t) for t in batch["labels"]]

            # forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack(
                [target["orig_size"] for target in labels], dim=0
            )
            results = im_processor.post_process(
                outputs, orig_target_sizes
            )  # convert outputs of model to COCO api

            module.add(prediction=results, reference=labels)
            del batch

    results = module.compute()
    return results


def get_dataset_image_ids(dataset: DatasetDict, dataset_type: str) -> list:
    # dataset_type is either train, test or eval
    image_id_list = []
    for example in dataset[dataset_type]:
        image_id_list.append(example["image_id"])
    return image_id_list


def get_filenames_by_ids(image_ids: list, annotations: dict, id_list) -> dict:
    id2filename = {key: "" for key in image_ids}
    for image_id in id_list:
        for element in annotations["images"]:
            if image_id == element["id"]:
                id2filename[image_id] = element["file_name"]
    return id2filename

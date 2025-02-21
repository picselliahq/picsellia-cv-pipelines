import os

import torch
import transformers
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.label import Label
from PIL import Image


def download_data(experiment: Experiment) -> DatasetVersion:
    dataset_list = experiment.list_attached_dataset_versions()
    dataset = dataset_list[0]
    dataset.download(os.path.join(experiment.base_dir, "data"))

    return dataset


def evaluate_asset(
    file_path: str,
    data_dir: str,
    experiment: Experiment,
    model: transformers.models,
    image_processor: transformers.models,
    dataset: DatasetVersion,
):
    dataset_labels = {label.name: label for label in dataset.list_labels()}
    image_path = os.path.join(data_dir, file_path)
    asset = find_asset_from_path(image_path=image_path, dataset=dataset)
    results = predict_image(
        image_path=image_path,
        threshold=0.4,
        model=model,
        image_processor=image_processor,
    )
    rectangle_list = create_rectangle_list(
        results, dataset_labels, model.config.id2label
    )
    send_rectangle_list_to_evaluations(rectangle_list, experiment, asset)


def find_asset_from_path(image_path: str, dataset: DatasetVersion) -> Asset:
    asset_filename = get_filename_from_fullpath(image_path)
    try:
        asset = dataset.find_asset(filename=asset_filename)
    except Exception as e:
        print(e)
    return asset


def get_filename_from_fullpath(full_path: str) -> str:
    return full_path.split("/")[-1]


def predict_image(
    image_path: str, threshold: float, image_processor, model: transformers.models
) -> dict:
    with torch.no_grad():
        image = Image.open(image_path)
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        # box format in results is: top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    return results


def create_rectangle_list(
    results: dict, dataset_labels: dict, id2label: dict
) -> list[tuple[int, int, int, int, Label, float]]:
    rectangle_list = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        formatted_box = reformat_box_to_coco(box)
        score = round(score.item(), 3)
        detected_label = dataset_labels[id2label[label.item()]]

        rectangle_list.append(
            (
                formatted_box[0],
                formatted_box[1],
                formatted_box[2],
                formatted_box[3],
                detected_label,
                score,
            )
        )

    return rectangle_list


def reformat_box_to_coco(box: torch.Tensor) -> list[int]:
    box = [int(i) for i in box.tolist()]
    formatted_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    return formatted_box


def send_rectangle_list_to_evaluations(
    rectangle_list: list, experiment: Experiment, asset: Asset
):
    if len(rectangle_list) > 0:
        try:
            experiment.add_evaluation(asset=asset, rectangles=rectangle_list)
            print(f"Asset: {asset.filename} evaluated.")
        except Exception as e:
            print(e)
            pass

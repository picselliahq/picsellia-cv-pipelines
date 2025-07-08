import os

import yaml
from picsellia_cv_engine.core.data.dataset.dataset_collection import DatasetCollection
from picsellia_cv_engine.core.data.dataset.yolo_dataset import YoloDataset


def generate_data_yaml(
    picsellia_datasets: DatasetCollection[YoloDataset],
) -> str:
    data_yaml = {
        "train": os.path.join(picsellia_datasets.dataset_path, "images", "train"),
        "val": os.path.join(picsellia_datasets.dataset_path, "images", "val"),
        "test": os.path.join(picsellia_datasets.dataset_path, "images", "test"),
        "nc": len(picsellia_datasets["train"].labelmap.keys()),
        "names": list(picsellia_datasets["train"].labelmap.keys()),
    }

    with open(os.path.join(picsellia_datasets.dataset_path, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return os.path.join(picsellia_datasets.dataset_path, "data.yaml")

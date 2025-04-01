import os

import yaml
from picsellia_cv_engine import step
from picsellia_cv_engine.core import DatasetCollection, YoloDataset


@step
def prepare_ultralytics_dataset_collection(
    dataset_collection: DatasetCollection[YoloDataset],
) -> DatasetCollection[YoloDataset]:
    data_yaml = {
        "train": os.path.join(dataset_collection.dataset_path, "images", "train"),
        "val": os.path.join(dataset_collection.dataset_path, "images", "val"),
        "test": os.path.join(dataset_collection.dataset_path, "images", "test"),
        "nc": len(dataset_collection["train"].labelmap.keys()),
        "names": list(dataset_collection["train"].labelmap.keys()),
    }

    with open(os.path.join(dataset_collection.dataset_path, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return dataset_collection

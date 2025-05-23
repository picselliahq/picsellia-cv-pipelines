import os

import yaml
from picsellia_cv_engine.core import (
    DatasetCollection,
)
from picsellia_cv_engine.core.data import (
    TBaseDataset,
)


class Yolov7DatasetCollection(DatasetCollection[TBaseDataset]):
    def __init__(self, datasets: list[TBaseDataset]):
        super().__init__(datasets=datasets)
        self.config_path: str | None = None

    def write_config(self, config_path: str) -> None:
        """
        Writes the dataset collection configuration to a YAML file.

        Args:
            config_path (str): The path to the configuration file.
        """
        if not self.dataset_path:
            raise ValueError(
                "Dataset path is required to write the configuration file."
            )
        with open(config_path, "w") as f:
            data = {
                "train": os.path.join(self.dataset_path, "images", "train"),
                "val": os.path.join(self.dataset_path, "images", "val"),
                "test": os.path.join(self.dataset_path, "images", "test"),
                "nc": len(self["train"].labelmap),
                "names": list(self["train"].labelmap.keys()),
            }
            yaml.dump(data, f)
        self.config_path = config_path

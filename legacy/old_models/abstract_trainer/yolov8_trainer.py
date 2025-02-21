import os
from abc import abstractmethod

from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.types.enums import LogType
from pycocotools.coco import COCO

from abstract_trainer.trainer import AbstractTrainer
from core_utils.yolov8 import (
    extract_file_name,
    get_batch_mosaics,
    get_metrics_curves,
    get_prop_parameter,
    get_train_test_eval_datasets_from_experiment,
    log_split_dataset_repartition_to_experiment,
    make_annotation_dict_by_dataset,
    write_annotation_file,
)


class Yolov8Trainer(AbstractTrainer):
    def __init__(self):
        super().__init__()
        self.trainer = None
        self.evaluation_ds = None
        self.evaluation_assets = None
        self.annotations_dict = None
        self.annotations_coco = None
        self.label_names = None
        self.current_dir = os.path.join(os.getcwd(), self.experiment.base_dir)
        self.annotations_path = "annotations.json"

    def prepare_data_for_training(self):
        self.experiment.download_artifacts(with_tree=True)
        (
            has_three_datasets,
            has_two_datasets,
            self.train_set,
            self.test_set,
            self.eval_set,
        ) = get_train_test_eval_datasets_from_experiment(self.experiment)

        if self.train_set:
            labels = self.train_set.list_labels()
            self.label_names = [label.name for label in labels]
            self.labelmap = {str(i): label.name for i, label in enumerate(labels)}
            self.experiment.log("labelmap", self.labelmap, "labelmap", replace=True)
        if has_three_datasets:
            self._process_triple_dataset()
        if has_two_datasets:  # self.eval_set is NONE
            self._process_double_dataset()

        elif not has_three_datasets and not has_two_datasets:
            self._process_single_dataset()

    def _process_triple_dataset(self):
        for data_type, dataset in {
            "train": self.train_set,
            "test": self.test_set,
            "val": self.eval_set,
        }.items():
            self._prepare_coco_annotations(dataset=dataset)
            self.download_data_with_label(
                assets=dataset.list_assets(), data_type=data_type
            )
        self.evaluation_ds = self.test_set
        self.evaluation_assets = self.evaluation_ds.list_assets()

    def _process_double_dataset(self):
        self._prepare_coco_annotations(dataset=self.train_set)
        (
            train_assets,
            eval_assets,
            _,
            _,
            labels,
        ) = self.train_set.train_test_split(prop=0.8)

        test_assets = self.test_set.list_assets()
        for data_type, assets in {
            "train": train_assets,
            "val": eval_assets,
            "test": test_assets,
        }.items():
            self.download_data_with_label(assets=assets, data_type=data_type)

        self.evaluation_ds = self.test_set
        self.evaluation_assets = self.evaluation_ds.list_assets()

    def _process_single_dataset(self):
        self._prepare_coco_annotations(dataset=self.train_set)
        prop = get_prop_parameter(parameters=self.parameters)
        (
            train_assets,
            test_assets,
            eval_assets,
            train_rep,
            test_rep,
            val_rep,
            labels,
        ) = self.train_set.train_test_val_split(
            ([prop, (1 - prop) / 2, (1 - prop) / 2])
        )

        for data_type, assets in {
            "train": train_assets,
            "val": eval_assets,
            "test": test_assets,
        }.items():
            self.download_data_with_label(assets=assets, data_type=data_type)
        self.evaluation_ds = self.train_set
        self.evaluation_assets = test_assets
        log_split_dataset_repartition_to_experiment(
            experiment=self.experiment,
            labelmap=self.labelmap,
            train_rep=train_rep,
            test_rep=test_rep,
            val_rep=val_rep,
        )

    def _prepare_coco_annotations(self, dataset: DatasetVersion):
        self.annotations_dict = make_annotation_dict_by_dataset(
            dataset=dataset, label_names=self.label_names
        )
        write_annotation_file(
            annotations_dict=self.annotations_dict,
            annotations_path=self.annotations_path,
        )
        self.annotations_coco = COCO(self.annotations_path)

    @abstractmethod
    def download_data_with_label(self, assets: MultiAsset, data_type: str):
        pass

    def train(self):
        self.setup_trainer()
        self.launch_trainer()

    @abstractmethod
    def setup_trainer(self):
        pass

    def launch_trainer(self):
        self.trainer.train()
        self.send_run_to_picsellia()

    def send_run_to_picsellia(self):
        final_run_path = self.trainer.save_dir
        for curve in get_metrics_curves(final_run_path):
            if curve:
                name = extract_file_name(curve)
                self.experiment.log(name, curve, LogType.IMAGE)

        for batch in get_batch_mosaics(final_run_path):
            if batch:
                name = extract_file_name(batch)
                self.experiment.log(name, batch, LogType.IMAGE)

    @abstractmethod
    def eval(self):
        pass

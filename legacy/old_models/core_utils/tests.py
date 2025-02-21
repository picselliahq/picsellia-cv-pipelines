import json
import os
import shutil
import sys
import time
import unittest
from datetime import date
from unittest.mock import patch

import yaml
from picsellia import Client

sys.path.append(os.path.join(os.getcwd(), "core_utils"))

from yolov8 import (
    extract_file_name,
    find_final_run,
    generate_data_yaml,
    get_batch_mosaics,
    get_metrics_curves,
    get_prop_parameter,
    get_train_test_eval_datasets_from_experiment,
    get_weights_and_config,
    make_annotation_dict_by_dataset,
    order_repartition_according_labelmap,
    setup_hyp,
    write_annotation_file,
)

TOKEN = os.environ["api_token"]
ORGA_ID = os.environ["organization_id"]


class TestYoloUtils(unittest.TestCase):
    organization_id = None
    labels = None
    checkpoint_path = None
    train_set = None
    dataset = None
    experiment = None
    project = None
    client = None
    token: str
    annotations_path = "test_annotations.json"
    labelmap = {0: "classA", 1: "classB", 2: "classC"}
    cwd = os.getcwd()

    @classmethod
    def setUpClass(cls) -> None:
        cls.token = TOKEN
        cls.organization_id = ORGA_ID
        cls.client = Client(
            api_token=cls.token,
            organization_id=cls.organization_id,
            host="https://staging.picsellia.com/",
        )
        cls.project = cls.client.create_project(
            name=f"test_yolo-utils{str(date.today())}-{str(time.time())}"
        )
        cls.experiment = cls.project.create_experiment(name="yolo-single-dataset")
        cls.dataset = cls.client.get_dataset_by_id(
            "01892b85-3c3b-72c8-943d-8a780c46a82d"
        )  # segmentation dataset
        cls.train_set = cls.dataset.get_version("eval")
        cls.experiment.attach_dataset(name="full", dataset_version=cls.train_set)
        cls.current_dir = os.path.join(cls.cwd, cls.experiment.base_dir)
        cls.parameters = {}
        cls.labels = cls.train_set.list_labels()
        cls.label_names = [label.name for label in cls.labels]
        cls.checkpoint_path = os.path.join(cls.experiment.checkpoint_dir, "test.pt")
        os.makedirs(cls.experiment.checkpoint_dir, exist_ok=True)

        with open(cls.checkpoint_path, "w") as f:
            f.write("dummy content")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()
        if os.path.isfile(cls.annotations_path):
            os.remove(cls.annotations_path)
        if os.path.exists(os.path.join(cls.cwd, cls.experiment.base_dir)):
            shutil.rmtree(os.path.join(cls.cwd, cls.experiment.base_dir))

        if os.path.exists(os.path.join(cls.cwd, "runs")):
            shutil.rmtree(os.path.join(cls.cwd, "runs"))

    def test_get_train_test_eval_datasets_from_experiment(self):
        (
            has_three_datasets,
            has_two_datasets,
            train_set,
            test_set,
            eval_set,
        ) = get_train_test_eval_datasets_from_experiment(experiment=self.experiment)
        self.assertFalse(has_three_datasets)
        self.assertEqual(self.train_set, train_set)
        self.assertEqual(None, test_set)
        self.assertEqual(None, eval_set)

    def test_write_annotation_file(self):
        annotations_dict = {"key1": "value1", "key2": "value2"}
        annotations_path = self.annotations_path

        write_annotation_file(annotations_dict, annotations_path)
        self.assertTrue(os.path.isfile(annotations_path))
        with open(annotations_path, "r") as f:
            written_data = json.load(f)
        self.assertEqual(annotations_dict, written_data)

    def test_get_prop_parameters(self):
        parameters = {"epochs": 100, "patience": 500, "image_size": 640}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.7)

    def test_get_prop_parameters_exit(self):
        parameters = {"epochs": 100, "patience": 500, "prop_train_split": 0.8}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.8)

    def test_order_repartition(self):
        repartition = {"x": ["classA", "classC", "classB"], "y": [2, 0, 1]}
        ordered_rep = order_repartition_according_labelmap(self.labelmap, repartition)
        expected_ordered_rep = {"x": ["classA", "classB", "classC"], "y": [2, 1, 0]}

        self.assertEqual(expected_ordered_rep, ordered_rep)

    def test_order_repartition_missing_label(self):
        repartition = {"x": ["classC", "classB"], "y": [2, 1]}
        ordered_rep = order_repartition_according_labelmap(self.labelmap, repartition)
        expected_ordered_rep = {"x": ["classA", "classB", "classC"], "y": [0, 1, 2]}

        self.assertEqual(expected_ordered_rep, ordered_rep)

    def test_generate_data_yaml(self):
        data_yaml_path = generate_data_yaml(
            experiment=self.experiment,
            labelmap=self.labelmap,
            config_path=self.current_dir,
        )

        self.assertTrue(os.path.exists(self.current_dir))
        config_yaml_path = os.path.join(self.current_dir, "data_config.yaml")
        with open(config_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        expected_data_config = {
            "train": os.path.join(self.cwd, self.experiment.png_dir, "train"),
            "val": os.path.join(self.cwd, self.experiment.png_dir, "val"),
            "test": os.path.join(self.cwd, self.experiment.png_dir, "test"),
            "nc": len(self.labelmap),
            "names": sorted(self.labelmap.values()),
        }
        self.assertEqual(data_config, expected_data_config)

    def test_find_final_run(self):
        runs_path = os.path.join(self.cwd, "runs", "train")

        os.makedirs(os.path.join(runs_path, "run"), exist_ok=True)
        final_run_path = find_final_run(self.cwd)

        expected_path = os.path.join(runs_path, "run")
        self.assertEqual(expected_path, final_run_path)

        os.rmdir(os.path.join(runs_path, "run"))

    def setUp(self) -> None:
        self.test_dir = self.cwd
        self.weights_path = os.path.join(self.test_dir, "weights")
        self.best_path = os.path.join(self.weights_path, "best.pt")
        self.last_path = os.path.join(self.weights_path, "last.pt")
        self.yaml_hyp_path = os.path.join(self.test_dir, "hyp.yaml")
        self.args_path = os.path.join(self.test_dir, "args.yaml")
        self.final_run_path = self.test_dir
        self.create_weights_config_test_files()

    def tearDown(self) -> None:
        list_files_to_delete = [
            self.weights_path,
            self.yaml_hyp_path,
            self.args_path,
        ]
        for file in list_files_to_delete:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except IsADirectoryError:
                    shutil.rmtree(file)

    def create_weights_config_test_files(self):
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        with open(self.best_path, "w") as f:
            f.write("dummy best.pt content")
        with open(self.last_path, "w") as f:
            f.write("dummy last.pt content")
        with open(self.yaml_hyp_path, "w") as f:
            f.write("dummy hyp.yaml content")
        with open(self.args_path, "w") as f:
            f.write("dummy args.yaml content")

    def test_get_weights_and_config_with_best_weights(self):
        result = get_weights_and_config(self.final_run_path)
        self.assertEqual(
            (os.path.join(self.final_run_path, "weights", "best.pt"), self.args_path),
            result,
        )

    def test_get_weights_and_config_with_last(self):
        os.remove(self.best_path)
        result = get_weights_and_config(self.final_run_path)
        self.assertEqual((self.last_path, self.args_path), result)

    def test_get_weights_and_config_with_args_yaml(self):
        final_run_path = self.test_dir
        os.remove(self.best_path)
        os.remove(self.yaml_hyp_path)
        result = get_weights_and_config(final_run_path)
        self.assertEqual((self.last_path, self.args_path), result)

    def test_get_weights_and_config_with_no_files(self):
        final_run_path = self.test_dir
        os.remove(self.best_path)
        os.remove(self.args_path)
        os.remove(self.yaml_hyp_path)
        os.remove(self.last_path)
        result = get_weights_and_config(final_run_path)
        self.assertEqual((None, None), result)

    def test_extract_file_name_with_extension(self):
        file_path = "/path/to/file.txt"
        result = extract_file_name(file_path)
        self.assertEqual(result, "file")

    def test_make_annotation_dict_by_dataset(self):
        results = make_annotation_dict_by_dataset(
            dataset=self.train_set, label_names=self.label_names
        )
        self.assertTrue(dict, type(results))

    @patch("os.path.isfile")
    def test_get_metrics_curves_with_existing_files(self, mock_isfile):
        mock_isfile.return_value = True
        expected_result = (
            f"{self.final_run_path}/confusion_matrix.png",
            f"{self.final_run_path}/F1_curve.png",
            f"{self.final_run_path}/labels_correlogram.jpg",
            f"{self.final_run_path}/labels.jpg",
            f"{self.final_run_path}/P_curve.png",
            f"{self.final_run_path}/PR_curve.png",
            f"{self.final_run_path}/R_curve.png",
            f"{self.final_run_path}/BoxF1_curve.png",
            f"{self.final_run_path}/BoxP_curve.png",
            f"{self.final_run_path}/BoxPR_curve.png",
            f"{self.final_run_path}/BoxR_curve.png",
            f"{self.final_run_path}/MaskF1_curve.png",
            f"{self.final_run_path}/MaskP_curve.png",
            f"{self.final_run_path}/MaskPR_curve.png",
            f"{self.final_run_path}/MaskR_curve.png",
        )

        result = get_metrics_curves(self.final_run_path)
        self.assertEqual(result, expected_result)

    @patch("os.path.isfile")
    def test_get_metrics_curves_with_missing_files(self, mock_isfile):
        mock_isfile.return_value = False
        expected_result = tuple(None for _ in range(15))

        result = get_metrics_curves(self.final_run_path)
        self.assertEqual(result, expected_result)

    @patch("os.path.isfile")
    def test_get_batch_mosaics_with_existing_files(self, mock_isfile):
        mock_isfile.return_value = True
        final_run_path = "/path/to/final_run_directory"  # Replace with a valid path

        expected_result = (
            f"{final_run_path}/val_batch0_labels.jpg",
            f"{final_run_path}/val_batch0_pred.jpg",
            f"{final_run_path}/val_batch1_labels.jpg",
            f"{final_run_path}/val_batch1_pred.jpg",
            f"{final_run_path}/val_batch2_labels.jpg",
            f"{final_run_path}/val_batch2_pred.jpg",
        )

        result = get_batch_mosaics(final_run_path)
        self.assertEqual(expected_result, result)

    @patch("os.path.isfile", return_value=False)
    def test_get_batch_mosaics_with_missing_files(self, mock_isfile):
        mock_isfile.return_value = False
        expected_result = tuple(None for _ in range(6))

        result = get_batch_mosaics(self.final_run_path)
        self.assertEqual(expected_result, result)

    def test_setup_hyp(self):
        data_yaml_path = generate_data_yaml(
            experiment=self.experiment,
            labelmap=self.labelmap,
            config_path=self.current_dir,
        )

        config = setup_hyp(
            experiment=self.experiment,
            data_yaml_path=data_yaml_path,
            params=self.parameters,
            cwd=self.current_dir,
            task="segment",
        )
        self.assertEqual(config.task, "segment")
        self.assertEqual(config.mode, "train")


if __name__ == "__main__":
    unittest.main()

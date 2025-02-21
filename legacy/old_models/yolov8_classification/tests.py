import os
import shutil
import sys
import tempfile
import time
import unittest
from datetime import date

import numpy as np
from picsellia import Client

from core_utils.yolov8 import (
    _get_three_attached_datasets,
    get_train_test_eval_datasets_from_experiment,
)

sys.path.append(os.path.join(os.getcwd(), "yolov8-classification"))

from experiment.trainer import Yolov8ClassificationTrainer
from experiment.utils import (
    _create_class_directories,
    _create_coco_objects,
    format_confusion_matrix,
    get_image_annotation,
    get_prop_parameter,
    move_image,
    order_repartition_according_labelmap,
)

TOKEN = os.environ["api_token"]
ORGA_ID = os.environ["organization_id"]


class TestYolov8ClassificationUtils(unittest.TestCase):
    temp_dir = None
    organization_id = None
    base_path = None
    test_file_path = None
    eval_set = None
    test_set = None
    train_set = None
    dataset = None
    experiment = None
    project = None
    client = None
    token: str
    labelmap = {"0": "covid", "1": "normal", "2": "pneumonia"}

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
            name=f"test_yolo_classif-utils{str(date.today())}-{str(time.time())}"
        )
        cls.model_version = cls.client.get_model_version_by_id(
            "01894a84-42f1-7e0f-87a0-28d78d29ba81"
        )
        cls.experiment = cls.project.create_experiment(name="yolo-triple-dataset")
        cls.dataset = cls.client.get_dataset_by_id(
            "01892b88-e0bf-7fce-8f51-0dbab83eb094"
        )
        cls.experiment.attach_dataset(
            name="train", dataset_version=cls.dataset.get_version("train")
        )
        cls.experiment.attach_dataset(
            name="test", dataset_version=cls.dataset.get_version("test")
        )
        cls.experiment.attach_dataset(
            name="eval", dataset_version=cls.dataset.get_version("eval")
        )
        cls.train_set, cls.test_set, cls.eval_set = _get_three_attached_datasets(
            cls.experiment
        )
        cls.coco_train, cls.coco_test, cls.coco_val = _create_coco_objects(
            cls.train_set, cls.test_set, cls.eval_set
        )
        cls.base_path = os.path.join(os.getcwd(), "yolov8-classification")
        cls.train_path = os.path.join(cls.base_path, "data/train")
        cls.test_path = os.path.join(cls.base_path, "data/test")
        cls.val_path = os.path.join(cls.base_path, "data/val")

        cls.test_file_path = os.path.join(cls.base_path, "test_files")
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()
        sample_file_path = os.path.join(cls.base_path, "data/test/grape_image.jpeg")
        if os.path.isfile(sample_file_path):
            shutil.move(sample_file_path, cls.test_file_path)
        if os.path.exists(os.path.join(cls.base_path, "data")):
            shutil.rmtree(os.path.join(cls.base_path, "data"))
        shutil.rmtree(cls.temp_dir)
        if os.path.isfile("yolov8n.pt"):
            os.remove("yolov8n.pt")
        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def test_get_prop_parameters(self):
        parameters = {"epochs": 100, "patience": 500, "image_size": 640}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.7)

    def test_get_prop_parameters_exit(self):
        parameters = {"epochs": 100, "patience": 500, "prop_train_split": 0.8}
        prop = get_prop_parameter(parameters)
        self.assertEqual(prop, 0.8)

    def test_move_file(self):
        filename = "sample.txt"
        old_location_path = self.temp_dir
        new_location_path = os.path.join(self.temp_dir, "new_location")
        os.makedirs(new_location_path, exist_ok=True)
        old_path = os.path.join(old_location_path, filename)
        new_path = os.path.join(new_location_path, filename)
        with open(old_path, "w") as file:
            file.write("Sample content")

        move_image(filename, old_location_path, new_location_path)

        self.assertFalse(os.path.exists(old_path))
        self.assertTrue(os.path.exists(new_path))

    def test_create_class_directories(self):
        _create_class_directories(coco=self.coco_train, base_imdir=self.train_path)
        is_created = False
        if (
            os.path.exists(os.path.join(self.train_path, "covid"))
            and os.path.exists(os.path.join(self.train_path, "normal"))
            and os.path.exists(os.path.join(self.train_path, "pneumonia"))
        ):
            is_created = True
        self.assertTrue(is_created)

    def test_get_image_annotation(self):
        fnames = ["covid_082.jpg"]
        image = {
            "id": 0,
            "file_name": "covid_082.jpg",
            "width": 1165,
            "height": 1163,
            "license": None,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": None,
        }
        cat = get_image_annotation(coco=self.coco_train, fnames=fnames, image=image)
        expected_cat = {"id": 0, "name": "covid", "supercategory": None}

        self.assertEqual(expected_cat, cat)

    def test_get_three_attached_datasets(self):
        train_set, test_set, eval_test = _get_three_attached_datasets(self.experiment)
        self.assertEqual(
            (self.train_set, self.test_set, self.eval_set),
            (train_set, test_set, eval_test),
        )

    def test_get_train_eval_datasets_from_experiment(self):
        results = get_train_test_eval_datasets_from_experiment(self.experiment)
        expected_results = (True, False, self.train_set, self.test_set, self.eval_set)
        self.assertEqual(expected_results, results)

    def test_format_confusion_matrix(self):
        matrix = np.array([[60, 0, 0], [46, 0, 0], [39, 0, 0]])
        formatted_matrix = format_confusion_matrix(
            labelmap=self.labelmap, matrix=matrix
        )
        expected_matrix = {
            "categories": ["covid", "normal", "pneumonia"],
            "values": [[60, 0, 0], [46, 0, 0], [39, 0, 0]],
        }
        self.assertEqual(expected_matrix, formatted_matrix)

    def test_order_repartition_according_labelmap(self):
        repartition = {"x": ["covid", "pneumonia", "normal"], "y": [12, 11, 7]}
        ordered_repartition = order_repartition_according_labelmap(
            labelmap=self.labelmap,
            repartition=repartition,
        )
        expected_results = {"x": ["covid", "normal", "pneumonia"], "y": [12, 7, 11]}

        self.assertEqual(expected_results, ordered_repartition)


class TestYoloClassificationTrainer(unittest.TestCase):
    organization_id = None
    model_version = None
    dataset = None
    token: str
    experiment = None
    client = None
    project = None

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
            name=f"test_yolo_classif-helpers{str(date.today())}-{str(time.time())}"
        )
        cls.model_version = cls.client.get_model_version_by_id(
            "01894a84-42f1-7e0f-87a0-28d78d29ba81"
        )
        cls.experiment = cls.project.create_experiment(name="yolo-triple-dataset")
        cls.dataset = cls.client.get_dataset_by_id(
            "01892b88-e0bf-7fce-8f51-0dbab83eb094"
        )
        cls.experiment.attach_dataset(
            name="train", dataset_version=cls.dataset.get_version("train")
        )
        cls.experiment.log_parameters({"epochs": 1, "patience": 500, "image_size": 640})
        cls.experiment.attach_model_version(
            model_version=cls.model_version, do_attach_base_parameters=False
        )

        os.environ["experiment_id"] = str(cls.experiment.id)

        cls.training_pipeline = Yolov8ClassificationTrainer()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()
        shutil.rmtree("data")
        shutil.rmtree("images")
        shutil.rmtree("train")
        os.remove("yolov8x-cls.pt")

    def test_preprocess_train_test(self):
        self.training_pipeline.prepare_data_for_training()
        self.assertTrue(
            len(os.listdir(self.training_pipeline.val_folder_path)),
            len(self.training_pipeline.labelmap),
        )

        self.training_pipeline.train()
        self.assertTrue(os.path.isfile(self.training_pipeline.weights_path))
        self.assertTrue(
            os.path.isfile(
                os.path.join(self.training_pipeline.weights_dir_path, "last.onnx")
            )
        )

        self.training_pipeline.eval()
        self.assertTrue(
            isinstance(
                type(self.training_pipeline.experiment.get_log("confusion").data), dict
            )
        )

        evaluation_results = self.client.connexion.get(
            f"/sdk/experiment/{str(self.experiment.id)}/evaluations"
        ).json()
        self.assertGreater(evaluation_results["count"], 0)


if __name__ == "__main__":
    unittest.main()

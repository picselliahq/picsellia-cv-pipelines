import json
import os
import shutil
import sys
import time
import unittest
from datetime import date

from picsellia import Client
from pycocotools.coco import COCO

sys.path.append(os.path.join(os.getcwd(), "yolov8-segmentation"))
from experiment.trainer import Yolov8SegmentationTrainer
from experiment.utils import (
    coco_to_yolo_segmentation,
    create_img_label_segmentation,
    interleave_lists,
)

TOKEN = os.environ["api_token"]
ORGA_ID = os.environ["organization_id"]


class TestYolov8SegmentationUtils(unittest.TestCase):
    test_folder = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_folder = os.path.join(
            os.getcwd(),
            "yolov8-segmentation",
            "test_files",
        )

        cls.annotations_path_test = os.path.join(cls.test_folder, "annotations.json")

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_create_img_label_segmentation(self):
        with open(self.annotations_path_test) as json_file:
            annotations_dict = json.load(json_file)
        image = annotations_dict["images"][0]
        create_img_label_segmentation(
            image=image,
            annotations_coco=COCO(self.annotations_path_test),
            labels_path=self.test_folder,
            label_names=["banana"],
        )
        txt_name = os.path.splitext(image["file_name"])[0] + ".txt"
        txt_file_path = os.path.join(self.test_folder, txt_name)

        self.assertTrue(txt_file_path)
        self.assertEqual(
            2, self.get_number_lines_text_file(txt_file_path=txt_file_path)
        )
        os.remove(txt_file_path)

    @staticmethod
    def get_number_lines_text_file(txt_file_path: str) -> int:
        with open(txt_file_path, "r") as fp:
            return len(fp.readlines())

    def test_coco_to_yolo_segmentation(self):
        ann = [
            [50, 60, 70, 60, 70, 80, 50, 80],
        ]
        expected_results = [
            0.4166666666666667,
            0.5,
            0.5833333333333334,
            0.5,
            0.5833333333333334,
            0.6666666666666666,
            0.4166666666666667,
            0.6666666666666666,
        ]
        results = coco_to_yolo_segmentation(ann=ann, image_h=120, image_w=120)
        self.assertEqual(expected_results, results)

    def test_interleave_lists(self):
        xs = [10, 20, 30]
        ys = [5, 15, 25]
        result = interleave_lists(xs, ys)
        expected = [10, 5, 20, 15, 30, 25]
        self.assertEqual(result, expected)

    def test_interleave_lists_empty(self):
        xs = []
        ys = []
        result = interleave_lists(xs, ys)
        expected = []
        self.assertEqual(result, expected)


class TestYolov8SegmentationTrainer(unittest.TestCase):
    organization_id = None
    test_folder = None
    model_version = None
    checkpoint_path = None
    train_set = None
    dataset = None
    experiment = None
    project = None
    client = None
    token: str

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
        os.environ["experiment_id"] = str(cls.experiment.id)
        cls.model_version = cls.client.get_model_version_by_id(
            "01894a7f-d4b6-7bd7-be36-13398579329f"
        )

        cls.experiment.attach_model_version(
            model_version=cls.model_version, do_attach_base_parameters=False
        )
        cls.experiment.log_parameters(
            {
                "epochs": 2,
                "batch_size": 4,
                "image_size": 640,
                "save_period": 100,
                "prop_train_split": 0.7,
            }
        )

        cls.dataset = cls.client.get_dataset_by_id(
            "01892b85-3c3b-72c8-943d-8a780c46a82d"
        )  # segmentation dataset
        cls.train_set = cls.dataset.get_version("eval")
        cls.experiment.attach_dataset(name="full", dataset_version=cls.train_set)
        cls.asset_train_path = os.path.join(cls.experiment.png_dir, "train", "images")
        cls.asset_test_path = os.path.join(cls.experiment.png_dir, "test", "images")
        cls.asset_eval_path = os.path.join(cls.experiment.png_dir, "val", "images")

        cls.label_train_path = os.path.join(cls.experiment.png_dir, "train", "labels")
        cls.label_test_path = os.path.join(cls.experiment.png_dir, "test", "labels")
        cls.label_eval_path = os.path.join(cls.experiment.png_dir, "val", "labels")
        cls.checkpoint_path = os.path.join(cls.experiment.checkpoint_dir)
        cls.test_folder = os.path.join(
            os.getcwd(),
            "yolov8-segmentation",
            "test_files",
        )

        cls.annotations_path_test = os.path.join(cls.test_folder, "annotations.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()

        if os.path.exists(os.path.join(cls.experiment.base_dir)):
            shutil.rmtree(os.path.join(cls.experiment.base_dir))
        if os.path.exists("saved_model"):
            shutil.rmtree("saved_model")

    def test_yolov8_segmentation_trainer(self):
        yolov8_trainer = Yolov8SegmentationTrainer()
        self.assertNotEqual(None, yolov8_trainer)

        yolov8_trainer.prepare_data_for_training()
        self.assertTrue(os.path.isfile("annotations.json"))
        self.assert_model_files_downloaded()
        self.assert_assets_downloaded()
        self.assert_label_txt_files_created()
        self.assert_all_label_files_created()

        yolov8_trainer.setup_trainer()
        self.assertTrue(
            os.path.isfile(os.path.join(self.experiment.base_dir, "exp", "args.yaml"))
        )

        yolov8_trainer.launch_trainer()

        yolov8_trainer.eval()

        evaluation_results = self.client.connexion.get(
            f"/sdk/experiment/{str(self.experiment.id)}/evaluations"
        ).json()
        self.assertGreater(evaluation_results["count"], 0)

        self.assert_new_model_files_stored()
        self.assert_batch0_sent(final_run_path=yolov8_trainer.trainer.save_dir)

    def assert_model_files_downloaded(self):
        for filepath in [
            os.path.join(self.checkpoint_path, "default.yaml"),
            os.path.join(self.checkpoint_path, "yolov8m-seg.pt"),
            os.path.join(
                self.experiment.base_dir, "exported_model", "yolov8m-seg.onnx"
            ),
            os.path.join(self.experiment.config_dir, "yolov8m-seg.yaml"),
        ]:
            self.assertTrue(os.path.isfile(filepath))

    def assert_assets_downloaded(self):
        for directory_path in [
            self.asset_train_path,
            self.asset_test_path,
            self.asset_eval_path,
        ]:
            print(directory_path)
            self.assert_directory_exists_and_not_empty(directory_path=directory_path)

    def assert_label_txt_files_created(self):
        for directory_path in [
            self.label_train_path,
            self.label_test_path,
            self.label_eval_path,
        ]:
            self.assert_directory_exists_and_not_empty(directory_path=directory_path)

    def assert_all_label_files_created(self):
        image_to_text_file_dict = {
            self.asset_train_path: self.label_train_path,
            self.asset_test_path: self.label_test_path,
            self.asset_eval_path: self.label_eval_path,
        }
        for image_path, label_path in image_to_text_file_dict.items():
            image_files = os.listdir(image_path)
            label_files = os.listdir(label_path)
            self.assertEqual(len(image_files), len(label_files))

    def assert_new_model_files_stored(self):
        artifact_names = [
            artifact.filename for artifact in self.experiment.list_artifacts()
        ]
        self.assertTrue("args.yaml" in artifact_names)
        self.assertFalse("yolov8m-seg-onnx" in artifact_names)

    def assert_batch0_sent(self, final_run_path):
        val_batch0_labels_path = os.path.join(final_run_path, "val_batch0_labels.jpg")
        if os.path.isfile(val_batch0_labels_path):
            retrieved_log = self.experiment.get_log(name="val_batch0_labels")
            self.assertFalse(None, retrieved_log)

    def assert_directory_exists_and_not_empty(self, directory_path: str):
        self.assertTrue(
            os.path.exists(directory_path),
            f"Folder path '{directory_path}' does not exist",
        )
        self.assertTrue(len(os.listdir(directory_path)) > 0)


if __name__ == "__main__":
    unittest.main()

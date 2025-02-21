import json
import os
import shutil
import sys
import time
import unittest
from datetime import date

from picsellia import Client
from pycocotools.coco import COCO

sys.path.append(os.path.join(os.getcwd(), "yolov8-detection"))

from experiment.trainer import Yolov8DetectionTrainer
from experiment.utils import coco_to_yolo_detection, create_img_label_detection

TOKEN = os.environ["api_token"]
ORGA_ID = os.environ["organization_id"]


class TestYolov8DetectionUtils(unittest.TestCase):
    annotations_path_test = None
    test_folder = None

    @classmethod
    def setUpClass(cls) -> None:
        print(os.getcwd())
        cls.test_folder = os.path.join(os.getcwd(), "yolov8-detection", "test_files")
        cls.annotations_path_test = os.path.join(
            cls.test_folder,
            "annotations.json",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_create_img_label_detection(self):
        with open(self.annotations_path_test) as json_file:
            annotations_dict = json.load(json_file)
        image = annotations_dict["images"][0]
        create_img_label_detection(
            image=image,
            annotations_coco=COCO(self.annotations_path_test),
            labels_path=self.test_folder,
            label_names=["car"],
        )
        txt_name = os.path.splitext(image["file_name"])[0] + ".txt"
        txt_file_path = os.path.join(self.test_folder, txt_name)

        self.assertTrue(txt_file_path)

        self.assertEqual(
            16, self.get_number_lines_text_file(txt_file_path=txt_file_path)
        )
        os.remove(txt_file_path)

    @staticmethod
    def get_number_lines_text_file(txt_file_path: str) -> int:
        with open(txt_file_path, "r") as fp:
            return len(fp.readlines())

    def test_coco_to_yolo_detection(self):
        x1 = 342
        y1 = 946
        w = 366
        h = 133
        results = coco_to_yolo_detection(
            x1=x1, y1=y1, w=w, h=h, image_w=1920, image_h=1000
        )
        expected_results = [0.2734375, 1.0125, 0.190625, 0.133]
        self.assertEqual(expected_results, results)


class TestYolov8DetectionTrainer(unittest.TestCase):
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
        cls.experiment = cls.project.create_experiment(
            name="yolo-detection-single-dataset"
        )
        os.environ["experiment_id"] = str(cls.experiment.id)
        cls.model_version = cls.client.get_model_version_by_id(
            "01894a82-a59d-7d38-8eb0-a017329d8030"
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
            "01892b88-63ec-767f-91f9-0765e68d0f78"
        )
        cls.train_set = cls.dataset.get_version("single-class-test")
        cls.experiment.attach_dataset(name="full", dataset_version=cls.train_set)
        cls.asset_train_path = os.path.join(cls.experiment.png_dir, "train", "images")
        cls.asset_test_path = os.path.join(cls.experiment.png_dir, "test", "images")
        cls.asset_eval_path = os.path.join(cls.experiment.png_dir, "val", "images")

        cls.label_train_path = os.path.join(cls.experiment.png_dir, "train", "labels")
        cls.label_test_path = os.path.join(cls.experiment.png_dir, "test", "labels")
        cls.label_eval_path = os.path.join(cls.experiment.png_dir, "val", "labels")
        cls.checkpoint_path = os.path.join(cls.experiment.checkpoint_dir)
        cls.test_folder = os.path.join(os.getcwd(), "yolov8-detection", "test_files")
        cls.annotations_path_test = os.path.join(
            cls.test_folder,
            "annotations.json",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.project.delete()
        if os.path.exists(os.path.join(cls.experiment.base_dir)):
            shutil.rmtree(os.path.join(cls.experiment.base_dir))
        if os.path.exists("saved_model"):
            shutil.rmtree("saved_model")
        if os.path.exists("runs"):
            shutil.rmtree("runs")

    def test_yolov8_detection_trainer(self):
        yolov8_trainer = Yolov8DetectionTrainer()
        self.assertNotEqual(None, yolov8_trainer)

        yolov8_trainer.prepare_data_for_training()
        self.assertTrue(os.path.isfile(self.annotations_path_test))
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
            os.path.join(self.checkpoint_path, "yolov8m.pt"),
            os.path.join(self.experiment.base_dir, "exported_model", "yolov8m.onnx"),
            os.path.join(self.experiment.config_dir, "yolov8m.yaml"),
        ]:
            self.assertTrue(
                os.path.isfile(filepath),
                f"file path '{filepath}' does not exist",
            )

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

    def assert_directory_exists_and_not_empty(self, directory_path: str):
        self.assertTrue(
            os.path.exists(directory_path),
            f"Folder path '{directory_path}' does not exist",
        )
        self.assertTrue(len(os.listdir(directory_path)) > 0)

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


if __name__ == "__main__":
    unittest.main()

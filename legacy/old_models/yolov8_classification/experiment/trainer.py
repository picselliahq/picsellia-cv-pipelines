import os
import sys

sys.path.append(os.path.join(os.getcwd(), "yolov8-classification", "experiment"))
from picsellia.types.enums import AnnotationFileType
from pycocotools.coco import COCO
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
from utils import (
    create_and_log_labelmap,
    download_triple_dataset,
    log_confusion_to_experiment,
    log_split_dataset_repartition_to_experiment,
    make_train_test_val_dirs,
    move_all_files_in_class_directories,
    move_files_in_class_directories,
    move_images_in_train_val_folders,
    predict_evaluation_images,
    prepare_datasets_with_annotation,
    split_single_dataset,
)

from abstract_trainer.trainer import AbstractTrainer
from core_utils.yolov8 import get_train_test_eval_datasets_from_experiment
from evaluator.yolo_evaluator import ClassificationYOLOEvaluator


class Yolov8ClassificationTrainer(AbstractTrainer):
    def __init__(self):
        super().__init__()
        self.weights = self.experiment.get_artifact("weights")
        self.weights.download()
        self.model = YOLO(self.weights.filename)
        self.working_directory = os.getcwd()
        self.data_path = os.path.join(self.working_directory, "data")
        self.val_folder_path = os.path.join(self.data_path, "val")
        self.weights_dir_path = os.path.join(self.working_directory, "train", "weights")
        self.weights_path = os.path.join(self.weights_dir_path, "last.pt")
        self.evaluation_ds, self.evaluation_assets = None, None

    def prepare_data_for_training(self):
        (
            has_three_datasets,
            has_two_datasets,
            self.train_set,
            self.test_set,
            self.eval_set,
        ) = get_train_test_eval_datasets_from_experiment(experiment=self.experiment)
        if has_three_datasets:
            self._process_triple_dataset()

        if has_two_datasets:
            self._process_double_dataset()
        else:
            self._process_single_dataset()

        self.labelmap = create_and_log_labelmap(experiment=self.experiment)

    def _process_triple_dataset(self):
        download_triple_dataset(self.train_set, self.test_set, self.eval_set)
        _, _ = prepare_datasets_with_annotation(
            self.train_set, self.test_set, self.eval_set
        )
        self.evaluation_ds = self.eval_set
        self.evaluation_assets = self.evaluation_ds.list_assets()

    def _process_double_dataset(self):
        self.train_set.download("images")
        self.test_set.download(target_path=os.path.join("data", "test"), max_workers=8)
        (
            train_assets,
            eval_assets,
            _,
            _,
            labels,
        ) = self.train_set.train_test_split(prop=0.8)

        make_train_test_val_dirs()
        move_images_in_train_val_folders(train_assets, eval_assets)

        self._move_files_in_class_directories_double()
        self.evaluation_ds = self.test_set
        self.evaluation_assets = self.evaluation_ds.list_assets()

    def _move_files_in_class_directories_double(self) -> None:
        train_annotation_path = self.train_set.export_annotation_file(
            AnnotationFileType.COCO
        )
        eval_annotation_path = self.test_set.export_annotation_file(
            AnnotationFileType.COCO
        )
        coco_train = COCO(train_annotation_path)
        coco_eval = COCO(eval_annotation_path)
        move_files_in_class_directories(coco_train, "data/train")
        move_files_in_class_directories(coco_train, "data/val")
        move_files_in_class_directories(coco_eval, "data/test")

    def _process_single_dataset(self):
        self.train_set.download("images")
        (
            train_assets,
            test_assets,
            eval_assets,
            train_rep,
            test_rep,
            val_rep,
            labels,
        ) = split_single_dataset(experiment=self.experiment, train_set=self.train_set)
        move_all_files_in_class_directories(train_set=self.train_set)
        log_split_dataset_repartition_to_experiment(
            experiment=self.experiment,
            labelmap=self.labelmap,
            train_rep=train_rep,
            test_rep=test_rep,
            val_rep=val_rep,
        )
        self.evaluation_ds = self.train_set
        self.evaluation_assets = eval_assets

    def train(self):
        self.model.train(
            data=self.data_path,
            epochs=self.parameters["epochs"],
            imgsz=self.parameters["image_size"],
            patience=self.parameters["patience"],
            project=self.working_directory,
        )
        self._save_weights()
        self._save_onnx_model()

    def _save_weights(self):
        self.experiment.store("weights", self.weights_path)

    def _save_onnx_model(self):
        export_model = YOLO(self.weights_path)
        onnx_path = os.path.join(self.weights_dir_path, "last.onnx")
        export_model.export(format="onnx")
        self.experiment.store("model-latest", onnx_path)

    def eval(self):
        self.model = YOLO(self.weights_path)
        metrics = self.model.val(data=self.data_path)
        accuracy = metrics.top1
        self.experiment.log("val/accuracy", float(accuracy), "value")

        ground_truths, predictions = predict_evaluation_images(
            labelmap=self.labelmap,
            val_folder_path=self.val_folder_path,
            model=self.model,
        )
        classification_report(
            ground_truths, predictions, target_names=self.labelmap.values()
        )
        matrix = confusion_matrix(ground_truths, predictions)
        log_confusion_to_experiment(
            experiment=self.experiment, labelmap=self.labelmap, matrix=matrix
        )
        self._run_picsellia_evaluation()

    def _run_picsellia_evaluation(self):
        yolo_evaluator = ClassificationYOLOEvaluator(
            experiment=self.experiment,
            dataset=self.evaluation_ds,
            asset_list=self.evaluation_assets,
            confidence_threshold=self.parameters.get("confidence_threshold", 0.1),
            weights_path=self.weights_path,
        )

        yolo_evaluator.evaluate()

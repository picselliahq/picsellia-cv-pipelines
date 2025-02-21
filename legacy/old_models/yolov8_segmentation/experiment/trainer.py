import os
import sys

from picsellia.sdk.asset import MultiAsset

from abstract_trainer.yolov8_trainer import Yolov8Trainer
from core_utils.yolov8 import (
    generate_data_yaml,
    setup_hyp,
)

sys.path.append(os.path.join(os.getcwd(), "yolov8-segmentation", "experiment"))

from segmentation_trainer import PicselliaSegmentationTrainer
from utils import create_yolo_segmentation_label

from evaluator.yolo_evaluator import SegmentationYOLOEvaluator


class Yolov8SegmentationTrainer(Yolov8Trainer):
    def __init__(self):
        super().__init__()
        self.final_run_path = None

    def download_data_with_label(self, assets: MultiAsset, data_type: str):
        assets.download(
            target_path=os.path.join(self.experiment.png_dir, data_type, "images"),
            max_workers=8,
        )
        create_yolo_segmentation_label(
            self.experiment,
            data_type,
            self.annotations_dict,
            self.annotations_coco,
            self.label_names,
        )

    def setup_trainer(self):
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

        self.trainer = PicselliaSegmentationTrainer(
            experiment=self.experiment, cfg=config, parameters=self.parameters
        )

    def eval(self):
        segmentation_evaluator = SegmentationYOLOEvaluator(
            experiment=self.experiment,
            dataset=self.evaluation_ds,
            asset_list=self.evaluation_assets,
            confidence_threshold=self.parameters.get("confidence_threshold", 0.1),
        )

        segmentation_evaluator.evaluate()

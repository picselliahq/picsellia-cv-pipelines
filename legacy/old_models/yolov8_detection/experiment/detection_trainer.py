import logging
import os
from math import isnan

from picsellia import Experiment
from picsellia.types.enums import LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from core_utils.yolov8 import store_model_files


class PicselliaDetectionTrainer(DetectionTrainer):
    def __init__(self, experiment: Experiment, cfg=None, parameters: dict = None):
        args = dict(
            task=cfg.task,
            model=cfg.model or "yolov8n.pt",
            data=cfg.data or "coco128.yaml",  # or yolo.ClassificationDataset("mnist")
            device=cfg.device if cfg.device is not None else "",
            epochs=cfg.epochs,
            batch=cfg.batch,
            imgsz=cfg.imgsz,
            save_period=cfg.save_period,
            project=cfg.project,
            patience=cfg.patience,
            name=cfg.name,
        )
        cfg_dict = vars(cfg)
        for parameter_key in parameters.keys():
            if parameter_key in cfg_dict:
                parameter_type = type(cfg_dict[parameter_key])
                args[parameter_key] = parameter_type(parameters[parameter_key])

        super().__init__(overrides=args)
        self.experiment = experiment
        self.cwd = cfg.cwd
        self.epochs = cfg.epochs

    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        model = None
        for name, value in metrics.items():
            log_name = str(name).replace("/", "_")
            if not isinstance(value, (int, float)) or isnan(value):
                continue
            self._log_metric(log_name, float(value), retry=1)
        if self.epoch == self.epochs - 1:
            store_model_files(
                experiment=self.experiment,
                save_dir=self.save_dir,
                task="detect",
            )
        if (
            (self.epoch > 1)
            and (self.save_period > 0)
            and ((self.epoch - 1) % self.save_period == 0)
        ):
            try:
                model = YOLO(os.path.join(self.save_dir, "weights", "best.pt"))
            except FileNotFoundError:
                try:
                    model = YOLO(os.path.join(self.save_dir, "weights", "last.pt"))
                except FileNotFoundError as e:
                    logging.warning(
                        "Can't find intermediary checkpoint at save period, they will be uploaded at the end"
                    )
                    logging.warning(e)
            if model:
                model.export(format="onnx", imgsz=self.args.imgsz, task="detect")
                store_model_files(
                    experiment=self.experiment,
                    save_dir=self.save_dir,
                    task="detect",
                )

    def _log_metric(self, name: str, value: float, retry: int):
        if not isnan(value):
            try:
                self.experiment.log(name=name, type=LogType.LINE, data=value)
            except Exception as e:
                logging.exception(f"couldn't log {name}")
                logging.warning(e)
                if retry > 0:
                    logging.info(f"retrying log {name}")
                    self._log_metric(name, value, retry - 1)

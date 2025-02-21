from src.picsellia_cv_engine.models.parameters.hyper_parameters import (
    HyperParameters,
)

from picsellia.types.schemas import LogDataType


class Yolov7HyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="0"
        )
        self.lr0 = self.extract_parameter(
            keys=["lr0"], expected_type=float, default=0.01
        )
        self.lrf = self.extract_parameter(
            keys=["lrf"], expected_type=float, default=0.1
        )
        self.momentum = self.extract_parameter(
            keys=["momentum"], expected_type=float, default=0.937
        )
        self.weight_decay = self.extract_parameter(
            keys=["weight_decay"], expected_type=float, default=0.0005
        )
        self.warmup_epochs = self.extract_parameter(
            keys=["warmup_epochs"], expected_type=float, default=3.0
        )
        self.warmup_momentum = self.extract_parameter(
            keys=["warmup_momentum"], expected_type=float, default=0.8
        )
        self.warmup_bias_lr = self.extract_parameter(
            keys=["warmup_bias_lr"], expected_type=float, default=0.1
        )
        self.box = self.extract_parameter(
            keys=["box_loss_gain"], expected_type=float, default=0.05
        )
        self.cls = self.extract_parameter(
            keys=["cls_loss_gain"], expected_type=float, default=0.3
        )
        self.cls_pw = self.extract_parameter(
            keys=["cls_bce_loss_positive_weight"], expected_type=float, default=1.0
        )
        self.obj = self.extract_parameter(
            keys=["obj_loss_gain"], expected_type=float, default=0.7
        )
        self.obj_pw = self.extract_parameter(
            keys=["obj_bce_loss_positive_weight"], expected_type=float, default=1.0
        )
        self.iou_t = self.extract_parameter(
            keys=["iou_threshold"], expected_type=float, default=0.20
        )
        self.anchor_t = self.extract_parameter(
            keys=["anchor_threshold"], expected_type=float, default=4.0
        )
        self.fl_gamma = self.extract_parameter(
            keys=["focal_loss_gamma"], expected_type=float, default=0.0
        )
        self.loss_ota = self.extract_parameter(
            keys=["loss_ota"], expected_type=int, default=1
        )
        self.confidence_threshold = self.extract_parameter(
            keys=["confidence_threshold"], expected_type=float, default=0.1
        )
        self.iou_threshold = self.extract_parameter(
            keys=["iou_threshold"], expected_type=float, default=0.45
        )

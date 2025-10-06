from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters.augmentation_parameters import (
    AugmentationParameters,
)
from picsellia_cv_engine.core.parameters.hyper_parameters import (
    HyperParameters,
)


class Yolov7HyperParameters(HyperParameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.patience = self.extract_parameter(
            keys=["patience"], expected_type=int, default=100
        )
        self.save_period = self.extract_parameter(
            keys=["save_period"], expected_type=int, default=100
        )
        self.close_mosaic = self.extract_parameter(
            keys=["close_mosaic"], expected_type=int, default=0
        )
        self.export_format = self.extract_parameter(
            keys=["export_format"], expected_type=str, default="onnx"
        )
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


class Yolov7AugmentationParameters(AugmentationParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        self.hsv_h = self.extract_parameter(
            keys=["hsv_h"], expected_type=float, default=0.015, range_value=(0.0, 1.0)
        )
        self.hsv_s = self.extract_parameter(
            keys=["hsv_s"], expected_type=float, default=0.7, range_value=(0.0, 1.0)
        )
        self.hsv_v = self.extract_parameter(
            keys=["hsv_v"], expected_type=float, default=0.4, range_value=(0.0, 1.0)
        )
        self.degrees = self.extract_parameter(
            keys=["degrees"],
            expected_type=float,
            default=0.0,
            range_value=(-180.0, 180.0),
        )
        self.translate = self.extract_parameter(
            keys=["translate"], expected_type=float, default=0.2, range_value=(0.0, 1.0)
        )
        self.scale = self.extract_parameter(
            keys=["scale"],
            expected_type=float,
            default=0.5,
            range_value=(
                0.0,
                float("inf"),
            ),
        )
        self.shear = self.extract_parameter(
            keys=["shear"],
            expected_type=float,
            default=0.0,
            range_value=(-180.0, 180.0),
        )
        self.perspective = self.extract_parameter(
            keys=["perspective"],
            expected_type=float,
            default=0.0,
            range_value=(0.0, 0.001),
        )
        self.flipud = self.extract_parameter(
            keys=["flipud"], expected_type=float, default=0.0, range_value=(0.0, 1.0)
        )
        self.fliplr = self.extract_parameter(
            keys=["fliplr"], expected_type=float, default=0.5, range_value=(0.0, 1.0)
        )
        self.mosaic = self.extract_parameter(
            keys=["mosaic"], expected_type=float, default=1.0, range_value=(0.0, 1.0)
        )
        self.mixup = self.extract_parameter(
            keys=["mixup"], expected_type=float, default=0.0, range_value=(0.0, 1.0)
        )
        self.copy_paste = self.extract_parameter(
            keys=["copy_paste"],
            expected_type=float,
            default=0.0,
            range_value=(0.0, 1.0),
        )
        self.paste_in = self.extract_parameter(
            keys=["paste_in"],
            expected_type=float,
            default=0.0,
            range_value=(0.0, 1.0),
        )

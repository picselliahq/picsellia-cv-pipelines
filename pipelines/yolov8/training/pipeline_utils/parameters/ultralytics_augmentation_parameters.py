from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.models.parameters.augmentation_parameters import (
    AugmentationParameters,
)


class UltralyticsAugmentationParameters(AugmentationParameters):
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
            keys=["translate"], expected_type=float, default=0.1, range_value=(0.0, 1.0)
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
        self.bgr = self.extract_parameter(
            keys=["bgr"], expected_type=float, default=0.0, range_value=(0.0, 1.0)
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
        self.auto_augment = self.extract_parameter(
            keys=["auto_augment"], expected_type=str, default="randaugment"
        )
        self.erasing = self.extract_parameter(
            keys=["erasing"], expected_type=float, default=0.4, range_value=(0.0, 1.0)
        )
        self.crop_fraction = self.extract_parameter(
            keys=["crop_fraction"],
            expected_type=float,
            default=1.0,
            range_value=(0.1, 1.0),
        )

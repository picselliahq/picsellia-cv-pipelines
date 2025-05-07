from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters.augmentation_parameters import (
    AugmentationParameters,
)


class PaddleOCRAugmentationParameters(AugmentationParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

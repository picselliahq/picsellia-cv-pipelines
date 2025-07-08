from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import HyperParameters


class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.epochs = self.extract_parameter(["epochs"], expected_type=int, default=3)
        self.batch_size = self.extract_parameter(
            ["batch_size"], expected_type=int, default=8
        )
        self.image_size = self.extract_parameter(
            ["image_size"], expected_type=int, default=640
        )

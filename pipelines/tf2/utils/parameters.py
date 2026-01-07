from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import HyperParameters


class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.steps = self.extract_parameter(["steps"], expected_type=int, default=3)
        self.batch_size = self.extract_parameter(["batch_size"], expected_type=int, default=8)
        self.image_size = self.extract_parameter(["image_size"], expected_type=int, default=640)
        self.learning_rate = self.extract_parameter(["learning_rate"], expected_type=float, default=0.0005)
        self.annotation_type = self.extract_parameter(["annotation_type"], expected_type=str, default="polygon") # ou rectangle
        self.checkpoint_every_n = self.extract_parameter(["checkpoint_every_n"], expected_type=int, default=10)

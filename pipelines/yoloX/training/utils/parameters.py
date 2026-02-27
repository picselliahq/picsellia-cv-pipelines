from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import ExportParameters, HyperParameters


class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.architecture = self.extract_parameter(
            ["architecture", "model_architecture"],
            expected_type=str,
            default="yolox-s",
        )
        self.epochs = self.extract_parameter(["epochs"], expected_type=int, default=100)
        self.batch_size = self.extract_parameter(
            ["batch_size"], expected_type=int, default=8
        )
        self.image_size = self.extract_parameter(
            ["image_size"], expected_type=int, default=640
        )
        self.learning_rate = self.extract_parameter(
            ["learning_rate", "lr"], expected_type=float, default=0.01 / 64
        )
        self.eval_interval = self.extract_parameter(
            ["eval_interval"], expected_type=int, default=5
        )
        self.enable_weather_transform = self.extract_parameter(
            ["enable_weather_transform"], expected_type=bool, default=False
        )
        self.evaluation_batch_size = self.extract_parameter(
            ["evaluation_batch_size"], expected_type=int, default=8
        )
        self.transfer_learning = self.extract_parameter(
            ["transfer_learning"], expected_type=bool, default=False
        )


class YOLOXExportParameters(ExportParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.enable_dynamic_axis = self.extract_parameter(
            ["enable_dynamic_axis"], expected_type=bool, default=False
        )

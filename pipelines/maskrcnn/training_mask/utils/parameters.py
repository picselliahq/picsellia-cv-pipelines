from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import ExportParameters, HyperParameters


class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.epochs = self.extract_parameter(["epochs"], expected_type=int, default=10)
        self.batch_size = self.extract_parameter(
            ["batch_size"], expected_type=int, default=4
        )
        self.image_size = self.extract_parameter(
            ["image_size"], expected_type=int, default=800
        )
        self.backbone = self.extract_parameter(
            ["backbone", "model_backbone"],
            expected_type=str,
            default="resnet50",
        )
        self.learning_rate = self.extract_parameter(
            ["learning_rate", "lr"], expected_type=float, default=5e-4
        )
        self.weight_decay = self.extract_parameter(
            ["weight_decay"], expected_type=float, default=5e-4
        )
        self.momentum = self.extract_parameter(
            ["momentum"], expected_type=float, default=0.9
        )
        self.lr_scheduler_step_size = self.extract_parameter(
            ["lr_scheduler_step_size", "step_size"], expected_type=int, default=5
        )
        self.lr_scheduler_gamma = self.extract_parameter(
            ["lr_scheduler_gamma", "gamma"], expected_type=float, default=0.1
        )
        self.transfer_learning = self.extract_parameter(
            ["transfer_learning"], expected_type=bool, default=False
        )


class MaskRCNNExportParameters(ExportParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.export_format = self.extract_parameter(
            keys=["export_format"], expected_type=str, default="torchscript"
        )

from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import HyperParameters


class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.epochs = self.extract_parameter(
            ["epochs", "epoch"], expected_type=int, default=3
        )
        self.batch_size = self.extract_parameter(
            ["batch_size"], expected_type=int, default=8
        )
        self.learning_rate = self.extract_parameter(
            ["learning_rate", "lr"], expected_type=float, default=0.00005
        )
        self.warmup_steps = self.extract_parameter(
            ["warmup_steps"], expected_type=int, default=0
        )
        self.weight_decay = self.extract_parameter(
            ["weight_decay"], expected_type=float, default=0.1
        )
        self.model_name = self.extract_parameter(
            ["model_name", "repo_id"],
            expected_type=str,
            default="openai/clip-vit-large-patch14-336",
        )
        self.caption_prompt = self.extract_parameter(
            ["caption_prompt"], expected_type=str, default="Describe the image"
        )

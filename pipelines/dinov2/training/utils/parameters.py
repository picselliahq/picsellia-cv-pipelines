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
        self.hugging_face_model_name = self.extract_parameter(
            ["hugging_face_model_name"],
            expected_type=str,
            default="facebook/dinov2-small",
        )
        self.patience = self.extract_parameter(
            ["patience"], expected_type=int, default=5
        )
        self.n_blocks = self.extract_parameter(
            ["n_blocks"], expected_type=int, default=3
        )
        self.use_bbox_features = self.extract_parameter(
            ["use_bbox_features"], expected_type=bool, default=True
        )

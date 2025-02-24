from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.models.parameters.base_parameters import Parameters


class PaddleOCRHyperParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.train_set_split_ratio = self.extract_parameter(
            keys=["prop_train_split", "train_set_split_ratio"],
            expected_type=float,
            default=0.8,
        )
        self.seed = self.extract_parameter(keys=["seed"], expected_type=int, default=0)
        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cuda:0"
        )
        self.bbox_epochs = self.extract_parameter(
            keys=["bbox/epoch"], expected_type=int, default=100
        )
        self.text_epochs = self.extract_parameter(
            keys=["text/epoch"], expected_type=int, default=100
        )
        self.bbox_batch_size = self.extract_parameter(
            keys=["bbox/batch_size"], expected_type=int, default=8
        )
        self.text_batch_size = self.extract_parameter(
            keys=["text/batch_size"], expected_type=int, default=8
        )
        self.bbox_learning_rate = self.extract_parameter(
            keys=["bbox/learning_rate"], expected_type=float, default=0.001
        )
        self.text_learning_rate = self.extract_parameter(
            keys=["text/learning_rate"], expected_type=float, default=0.001
        )
        self.bbox_save_epoch_step = self.extract_parameter(
            keys=["bbox/save_epoch_step"], expected_type=int, default=10
        )
        self.text_save_epoch_step = self.extract_parameter(
            keys=["text/save_epoch_step"], expected_type=int, default=10
        )
        self.max_text_length = self.extract_parameter(
            keys=["max_text_length"], expected_type=int, default=25
        )

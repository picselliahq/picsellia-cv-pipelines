from picsellia_cv_engine.models.parameters.base_parameters import Parameters


class ProcessingYOLOV8PreannotationParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.model_file_name = self.extract_parameter(
            keys=["model_file_name"], expected_type=str
        )
        self.confidence_threshold = self.extract_parameter(
            keys=["confidence_threshold"], expected_type=float
        )
        self.batch_size = self.extract_parameter(
            keys=["batch_size"], expected_type=int, default=8
        )
        self.image_size = self.extract_parameter(
            keys=["image_size"], expected_type=int, default=640
        )
        self.label_matching_strategy = self.extract_parameter(
            keys=["label_matching_strategy"], expected_type=str, default="exact"
        )
        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cuda"
        )

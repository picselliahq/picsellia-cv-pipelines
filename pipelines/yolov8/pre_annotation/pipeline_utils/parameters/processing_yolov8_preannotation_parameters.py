from picsellia_cv_engine.core.parameters.base_parameters import Parameters


class ProcessingYOLOV8PreannotationParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.model_file_name = self.extract_parameter(
            keys=["model_file_name"], expected_type=str, default="pretrained-weights"
        )
        self.confidence_threshold = self.extract_parameter(
            keys=["confidence_threshold"], expected_type=float, default=0.1
        )
        self.batch_size = self.extract_parameter(
            keys=["batch_size"], expected_type=int, default=8
        )
        self.image_size = self.extract_parameter(
            keys=["image_size"], expected_type=int, default=640
        )
        self.label_matching_strategy = self.extract_parameter(
            keys=["label_matching_strategy"], expected_type=str, default="add"
        )
        self.device = self.extract_parameter(
            keys=["device"], expected_type=str, default="cuda"
        )
        self.agnostic_nms = self.extract_parameter(
            keys=["agnostic_nms"], expected_type=bool, default=True
        )
        self.replace_annotations = self.extract_parameter(
            keys=["replace_annotations"], expected_type=bool, default=False
        )

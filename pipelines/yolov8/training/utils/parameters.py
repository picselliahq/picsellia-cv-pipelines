from picsellia_cv_engine.core.parameters import HyperParameters


class UltralyticsHyperParameters(HyperParameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.patience = self.extract_parameter(
            keys=["patience"], expected_type=int, default=100
        )
        self.save_period = self.extract_parameter(
            keys=["save_period"], expected_type=int, default=100
        )
        self.close_mosaic = self.extract_parameter(
            keys=["close_mosaic"], expected_type=int, default=0
        )
        self.export_format = self.extract_parameter(
            keys=["export_format"], expected_type=str, default="onnx"
        )

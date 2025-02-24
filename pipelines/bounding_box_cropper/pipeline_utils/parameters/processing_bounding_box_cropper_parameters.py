from picsellia_cv_engine.models.parameters.parameters import Parameters


class ProcessingBoundingBoxCropperParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.datalake = self.extract_parameter(
            keys=["datalake"], expected_type=str, default="default"
        )
        self.label_name_to_extract = self.extract_parameter(
            keys=["label_name_to_extract"], expected_type=str
        )
        self.data_tag = self.extract_parameter(keys=["data_tag"], expected_type=str)
        self.fix_annotation = self.extract_parameter(
            keys=["fix_annotation"], expected_type=bool, default=True
        )

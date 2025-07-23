from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.datalake = self.extract_parameter(
            ["datalake"], expected_type=str, default="default"
        )
        self.data_tag = self.extract_parameter(
            ["data_tag"], expected_type=str, default="processed"
        )
        self.mask_label = self.extract_parameter(
            ["mask_label"], expected_type=str, default="rack"
        )
        self.detection_dataset_id = self.extract_parameter(
            ["detection_dataset_id"],
            expected_type=str,
            default="019833ae-4310-75aa-922c-c5b3d24a0521",
        )

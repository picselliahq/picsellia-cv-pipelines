from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.datalake = self.extract_parameter(["datalake"], expected_type=str, default="default")
        self.data_tag = self.extract_parameter(["data_tag"], expected_type=str, default="processed")
        self.rotate_min = self.extract_parameter(
            ["rotate_min"], expected_type=int, default=-45
        )
        self.rotate_max = self.extract_parameter(
            ["rotate_max"], expected_type=int, default=45
        )
        self.scale_min = self.extract_parameter(
            ["scale_min"], expected_type=float, default=0.9
        )
        self.scale_max = self.extract_parameter(
            ["scale_max"], expected_type=float, default=1.1
        )
        self.rotate_prob = self.extract_parameter(
            ["rotate_prob"], expected_type=float, default=0.5
        )
        self.add_noise = self.extract_parameter(
            ["add_noise"], expected_type=bool, default=False
        )

from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.data_tag = self.extract_parameter(
            ["data_tag"], expected_type=str, default="processed"
        )
        self.fix_annotation = self.extract_parameter(
            ["fix_annotation"], expected_type=bool, default=True
        )

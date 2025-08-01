from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        self.box_threshold = self.extract_parameter(
            keys=["box_threshold"],
            expected_type=float,
            default=0.35,
            range_value=(0, 1),
        )
        self.text_threshold = self.extract_parameter(
            keys=["text_threshold"],
            expected_type=float,
            default=0.25,
            range_value=(0, 1),
        )

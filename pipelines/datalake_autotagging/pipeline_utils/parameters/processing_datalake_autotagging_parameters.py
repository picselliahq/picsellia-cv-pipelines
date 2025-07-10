import re

from picsellia_cv_engine.core.parameters.base_parameters import Parameters


class ProcessingDatalakeAutotaggingParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.tags_list: list[str] = re.findall(
            r"'(.*?)'", self.extract_parameter(keys=["tags_list"], expected_type=str)
        )
        self.batch_size = self.extract_parameter(
            keys=["batch_size"], expected_type=int, default=8
        )

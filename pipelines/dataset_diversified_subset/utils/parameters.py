from typing import Union

from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.n_samples = self.extract_parameter(
            ["n_samples"],
            expected_type=int,
            range_value=(1, float("inf")),
        )
        self.embedder_key = self.extract_parameter(
            ["embedder_key"],
            expected_type=Union[str, None],
            default=None,
        )
        self.with_annotations = self.extract_parameter(
            ["with_annotations"], expected_type=bool, default=False
        )
        self.with_tags = self.extract_parameter(
            ["with_tags"], expected_type=bool, default=False
        )
        self.seed = self.extract_parameter(["seed"], expected_type=int, default=0)

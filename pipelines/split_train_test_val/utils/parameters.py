from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.ratio_train = self.extract_parameter(["ratio_train"], expected_type=float, default=0.7)
        self.ratio_test = self.extract_parameter(["ratio_test"], expected_type=float, default=0.15)
        self.ratio_val = self.extract_parameter(["ratio_val"], expected_type=float, default=0.15)
        self.embed_asset_without_annotation = self.extract_parameter(["embed_asset_without_annotation"], expected_type=bool, default=True)


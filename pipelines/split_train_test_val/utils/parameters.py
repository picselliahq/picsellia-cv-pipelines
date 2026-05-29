from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.embed_asset_without_annotation = self.extract_parameter(["embed_asset_without_annotation"], expected_type=bool, default=True)
        self.add_asset_tags = self.extract_parameter(["add_asset_tags"], expected_type=bool, default=True)



from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.datalake = self.extract_parameter(
            ["datalake"], expected_type=str, default="default"
        )
        self.data_tag = self.extract_parameter(
            ["data_tag"], expected_type=str, default="video-frames"
        )
        self.frame_interval = self.extract_parameter(
            ["frame_interval"], expected_type=int, default=30
        )
        self.max_frames_per_video = self.extract_parameter(
            ["max_frames_per_video"], expected_type=int, default=0
        )

from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        # Picsellia's video annotation tool resizes videos to this fixed
        # canvas before display, but its COCO export does not rescale shape
        # coordinates back to the original video resolution.
        self.annotation_canvas_width = self.extract_parameter(
            keys=["annotation_canvas_width"],
            expected_type=int,
            default=2048,
            range_value=(1, 100_000),
        )
        self.annotation_canvas_height = self.extract_parameter(
            keys=["annotation_canvas_height"],
            expected_type=int,
            default=1150,
            range_value=(1, 100_000),
        )

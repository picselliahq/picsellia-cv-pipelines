from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        # If True, only extract frames marked as a track keyframe in
        # Picsellia's video annotation export (attributes.keyframe == True),
        # instead of every decoded frame. Mutually exclusive with
        # `extract_every_n_frames`.
        self.extract_only_keyframes = self.extract_parameter(
            keys=["extract_only_keyframes"],
            expected_type=bool,
            default=False,
        )

        # Extract one frame every N frames (0 = extract every frame).
        # Mutually exclusive with `extract_only_keyframes`.
        self.extract_every_n_frames = self.extract_parameter(
            keys=["extract_every_n_frames"],
            expected_type=int,
            default=0,
            range_value=(0, 100_000),
        )

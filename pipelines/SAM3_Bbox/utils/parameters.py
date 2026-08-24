from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        self.threshold = self.extract_parameter(
            ["threshold"], expected_type=float, default=0.5
        )

        self.mask_threshold = self.extract_parameter(
            ["mask_threshold"], expected_type=float, default=0.7
        )

        self.min_area = self.extract_parameter(
            ["min_area"], expected_type=float, default=50.0
        )

        self.max_overlap_ratio = self.extract_parameter(
            ["max_overlap_ratio"], expected_type=float, default=0.3
        )

        self.iou_threshold = self.extract_parameter(
            ["iou_threshold"], expected_type=float, default=0.5
        )

        self.containment_threshold = self.extract_parameter(
            ["containment_threshold"], expected_type=float, default=0.8
        )

        self.deduplication_strategy = self.extract_parameter(
            ["deduplication_strategy"], expected_type=str, default="keep_smaller"
        )

        # How to handle annotations already present on an asset when uploading
        # new ones: "keep" (do nothing), "replace" (delete and overwrite) or
        # "concatenate" (add new shapes on top of existing ones).
        self.annotation_mode = self.extract_parameter(
            ["annotation_mode"], expected_type=str, default="keep"
        )

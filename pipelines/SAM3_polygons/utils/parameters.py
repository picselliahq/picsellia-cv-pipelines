from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        # SAM-3 Segmentation Parameters

        # Detection confidence threshold (0.0 - 1.0)
        # Higher values = fewer but more confident detections
        self.threshold = self.extract_parameter(
            ["threshold"], expected_type=float, default=0.5
        )

        # Mask confidence threshold (0.0 - 1.0)
        # Higher values = tighter/smaller masks
        self.mask_threshold = self.extract_parameter(
            ["mask_threshold"], expected_type=float, default=0.7
        )

        # Post-processing parameters

        # Minimum area threshold in pixels
        # Polygons smaller than this will be filtered out
        self.min_area = self.extract_parameter(
            ["min_area"], expected_type=float, default=50.0
        )

        # Maximum allowed overlap ratio between polygons (0.0 - 1.0)
        # Polygons overlapping more than this ratio will be removed
        # The larger polygon is kept
        self.max_overlap_ratio = self.extract_parameter(
            ["max_overlap_ratio"], expected_type=float, default=0.3
        )

        # Multi-class deduplication parameters

        # IoU threshold for detecting overlapping masks (0.0 - 1.0)
        # Masks with IoU higher than this are considered duplicates
        self.iou_threshold = self.extract_parameter(
            ["iou_threshold"], expected_type=float, default=0.5
        )

        # Containment threshold for nested masks (0.0 - 1.0)
        # When one mask is contained within another above this ratio, they're duplicates
        self.containment_threshold = self.extract_parameter(
            ["containment_threshold"], expected_type=float, default=0.8
        )

        # Deduplication strategy: "keep_smaller" or "keep_larger"
        # keep_smaller: Prioritize smaller, more precise masks
        # keep_larger: Prioritize larger, more complete masks
        self.deduplication_strategy = self.extract_parameter(
            ["deduplication_strategy"], expected_type=str, default="keep_smaller"
        )

        # How to handle annotations already present on an asset when uploading
        # new ones: "keep" (do nothing), "replace" (delete and overwrite) or
        # "concatenate" (add new shapes on top of existing ones).
        self.annotation_mode = self.extract_parameter(
            ["annotation_mode"], expected_type=str, default="keep"
        )

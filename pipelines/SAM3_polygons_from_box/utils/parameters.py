from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import Parameters


class ProcessingParameters(Parameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        # SAM-3 Segmentation Parameters

        # Detection confidence threshold (0.0 - 1.0) applied to the mask
        # SAM-3 returns for each box prompt.
        self.threshold = self.extract_parameter(
            ["threshold"], expected_type=float, default=0.3
        )

        # Mask confidence threshold (0.0 - 1.0)
        # Higher values = tighter/smaller polygons
        self.mask_threshold = self.extract_parameter(
            ["mask_threshold"], expected_type=float, default=0.5
        )

        # Minimum polygon area threshold in pixels.
        # Masks smaller than this are discarded.
        self.min_area = self.extract_parameter(
            ["min_area"], expected_type=float, default=10.0
        )

        # If SAM-3 fails to produce a valid mask for a given box, fall back to
        # a rectangular polygon matching the original box so every input box
        # still has a corresponding output polygon.
        self.fallback_to_bbox_polygon = self.extract_parameter(
            ["fallback_to_bbox_polygon"], expected_type=bool, default=True
        )

        # How to handle annotations already present on the output dataset
        # version when uploading new ones: "keep" (do nothing), "replace"
        # (delete and overwrite) or "concatenate" (add new shapes on top).
        self.annotation_mode = self.extract_parameter(
            ["annotation_mode"], expected_type=str, default="replace"
        )

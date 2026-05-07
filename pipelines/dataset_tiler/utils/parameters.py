from enum import Enum

from picsellia_cv_engine.core.parameters import Parameters


class TileMode(Enum):
    CONSTANT = "constant"
    DROP = "drop"
    REFLECT = "reflect"
    EDGE = "edge"
    WRAP = "wrap"


class ProcessingTilerParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.overlap_height_ratio = self.extract_parameter(
            keys=["overlap_height_ratio"],
            expected_type=float,
            default=0.1,
            range_value=(0, 0.99),
        )
        self.overlap_width_ratio = self.extract_parameter(
            keys=["overlap_width_ratio"],
            expected_type=float,
            default=0.1,
            range_value=(0, 0.99),
        )
        self.min_annotation_area_ratio = self.extract_parameter(
            keys=["min_annotation_area_ratio", "min_area_ratio"],
            expected_type=float | None,
            default=0.0,
            range_value=(0, 0.99),
        )
        self.min_annotation_width = self.extract_parameter(
            keys=["min_annotation_width"],
            expected_type=int | None,
            default=0,
            range_value=(0, float("inf")),
        )
        self.min_annotation_height = self.extract_parameter(
            keys=["min_annotation_height"],
            expected_type=int | None,
            default=0,
            range_value=(0, float("inf")),
        )
        self.padding_color_value = self.extract_parameter(
            keys=["padding_color_value"],
            expected_type=int,
            default=114,
            range_value=(0, 255),
        )
        self.fix_annotation = self.extract_parameter(
            keys=["fix_annotation"], expected_type=bool, default=True
        )

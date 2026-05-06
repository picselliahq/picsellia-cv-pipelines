from enum import Enum

from picsellia.types.enums import ProcessingInputType
from picsellia_pipelines_cli.utils.inputs import PipelineInputs


class TileMode(Enum):
    CONSTANT = "constant"
    DROP = "drop"
    REFLECT = "reflect"
    EDGE = "edge"
    WRAP = "wrap"


class ProcessingInputs(PipelineInputs):
    def __init__(self):
        super().__init__()
        self.define_input(
            name="target_version_name",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )
        self.define_input(
            name="datalake",
            input_type=ProcessingInputType.DATALAKE,
            required=True,
        )
        self.define_input(
            name="data_tag",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )
        self.define_input(
            name="tile_height",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )
        self.define_input(
            name="tile_width",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )
        self.define_input(
            name="tiling_mode",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )

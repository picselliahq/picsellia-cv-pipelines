from picsellia.types.enums import ProcessingInputType
from picsellia_pipelines_cli.utils.inputs import PipelineInputs


class ProcessingInputs(PipelineInputs):
    def __init__(self):
        super().__init__()
        self.define_input(
            name="output_dataset_version_name",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )
        self.define_input(
            name="datalake_id",
            input_type=ProcessingInputType.DATALAKE,
            required=True,
        )
        self.define_input(
            name="frames_per_segment",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )

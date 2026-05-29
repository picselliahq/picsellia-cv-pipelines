from picsellia.types.enums import ProcessingInputType
from picsellia_pipelines_cli.utils.inputs import PipelineInputs


class ProcessingInputs(PipelineInputs):
    def __init__(self):
        super().__init__()
        self.define_input(
            name="model_version",
            input_type=ProcessingInputType.MODEL_VERSION,
            required=True,
        )
        self.define_input(
            name="model_file_name",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )
        self.define_input(
            name="confidence_threshold",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )

from picsellia.types.enums import ProcessingInputType
from picsellia_pipelines_cli.utils.inputs import PipelineInputs


class ProcessingInputs(PipelineInputs):
    def __init__(self):
        super().__init__()
        self.define_input(
            name="ratio_train",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )
        self.define_input(
            name="ratio_test",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )
        self.define_input(
            name="ratio_val",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )

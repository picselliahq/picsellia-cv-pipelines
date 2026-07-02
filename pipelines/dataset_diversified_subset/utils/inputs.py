from enum import Enum

from picsellia.types.enums import ProcessingInputType
from picsellia_pipelines_cli.utils.inputs import PipelineInputs


class DiversitySelectionAlgorithm(Enum):
    FPS = "fps"
    KMEANS = "kmeans"


class ProcessingInputs(PipelineInputs):
    def __init__(self):
        super().__init__()
        # Deliberately not named 'target_version_name': PicselliaDatasetProcessingContext
        # ._load_legacy_inputs() special-cases that exact key and auto-creates a dataset
        # version for it at context-init time, before this pipeline's own fork() step
        # runs — causing two dataset versions (one empty, one properly filled) to be
        # created for the same run. See shapes-cropper/pipeline.py and dataset_tiler/
        # pipeline.py for the same collision on other pipelines.
        self.define_input(
            name="new_version_name",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )
        self.define_input(
            name="algorithm",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )
        self.define_input(
            name="n_samples",
            input_type=ProcessingInputType.NUMBER,
            required=True,
        )

from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.diversified_dataset_extractor.pipeline_utils.steps_utils.data_validation.processing_diversified_data_extractor_data_validator import (
    ProcessingDiversifiedDataExtractorDataValidator,
)


@step
def validate_diversified_data_extractor_data(
    dataset: TBaseDataset,
) -> None:
    validator = ProcessingDiversifiedDataExtractorDataValidator(
        dataset=dataset,
    )
    validator.validate()

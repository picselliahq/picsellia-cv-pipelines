from diversified_dataset_extractor.pipeline_utils.parameters.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from diversified_dataset_extractor.pipeline_utils.steps.model_loading.processing_diversified_data_extractor_model_loader import (
    EmbeddingModel,
)
from diversified_dataset_extractor.pipeline_utils.steps_utils.processing.diversified_data_extractor_processing import (
    DiversifiedDataExtractorProcessing,
)
from picsellia_cv_engine.core.contexts.processing.dataset import (
    PicselliaDatasetProcessingContext,
)
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step


@step
def process(
    input_dataset: TBaseDataset,
    output_dataset: TBaseDataset,
    embedding_model: EmbeddingModel,
):
    context: PicselliaDatasetProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    processor = DiversifiedDataExtractorProcessing(
        client=context.client,
        datalake=context.client.get_datalake(),
        input_dataset=input_dataset,
        output_dataset_version=output_dataset.dataset_version,
        embedding_model=embedding_model,
        distance_threshold=context.processing_parameters.distance_threshold,
    )
    processor.process()

from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.diversified_dataset_extractor.pipeline_utils.parameters.processing_diversified_data_extractor_parameters import (
    ProcessingDiversifiedDataExtractorParameters,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps.model_loading.processing_diversified_data_extractor_model_loader import (
    EmbeddingModel,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps_utils.processing.diversified_data_extractor_processing import (
    DiversifiedDataExtractorProcessing,
)


@step
def process(dataset: TBaseDataset, embedding_model: EmbeddingModel):
    context: PicselliaProcessingContext[
        ProcessingDiversifiedDataExtractorParameters
    ] = Pipeline.get_active_context()

    processor = DiversifiedDataExtractorProcessing(
        client=context.client,
        datalake=context.client.get_datalake(),
        input_dataset=dataset,
        output_dataset_version=context.output_dataset_version,
        embedding_model=embedding_model,
        distance_threshold=context.processing_parameters.distance_threshold,
    )
    processor.process()

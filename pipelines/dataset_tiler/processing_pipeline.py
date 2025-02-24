from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.steps.data_extraction.processing_data_extractor import (
    get_processing_dataset_collection,
)
from picsellia_cv_engine.steps.data_upload.dataset_context_uploader import (
    upload_dataset_context,
)

from pipelines.dataset_tiler.pipeline_utils.parameters.processing_tiler_parameters import (
    ProcessingTilerParameters,
)
from pipelines.dataset_tiler.pipeline_utils.steps.data_validation.processing_tiler_data_validator import (
    validate_tiler_data,
)
from pipelines.dataset_tiler.pipeline_utils.steps.processing.tiler_processing import (
    process,
)


def get_context() -> PicselliaProcessingContext[ProcessingTilerParameters]:
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingTilerParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def tiler_processing_pipeline() -> None:
    dataset_collection = get_processing_dataset_collection()
    dataset_collection["input"] = validate_tiler_data(
        dataset_context=dataset_collection["input"]
    )
    output_dataset_context = process(dataset_collection=dataset_collection)
    upload_dataset_context(
        dataset_context=output_dataset_context,
        use_id=False,
        fail_on_asset_not_found=False,
    )


if __name__ == "__main__":
    tiler_processing_pipeline()

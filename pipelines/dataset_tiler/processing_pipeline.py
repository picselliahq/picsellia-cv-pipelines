from picsellia_cv_engine.core.contexts import PicselliaProcessingContext
from picsellia_cv_engine.core.steps.dataset.loader import load_coco_datasets
from picsellia_cv_engine.core.steps.dataset.uploader import upload_full_dataset
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline

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
    dataset_collection = load_coco_datasets()
    dataset_collection["input"] = validate_tiler_data(
        dataset=dataset_collection["input"]
    )
    output_dataset = process(dataset_collection=dataset_collection)
    upload_full_dataset(
        dataset=output_dataset,
        use_id=False,
        fail_on_asset_not_found=False,
    )


if __name__ == "__main__":
    tiler_processing_pipeline()

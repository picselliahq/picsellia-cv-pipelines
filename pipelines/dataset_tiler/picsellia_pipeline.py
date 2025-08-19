from picsellia_cv_engine.core.services.utils.picsellia_context import (
    create_picsellia_dataset_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset
from pipeline_utils.parameters.processing_tiler_parameters import (
    ProcessingTilerParameters,
)
from pipeline_utils.steps.data_validation.processing_tiler_data_validator import (
    validate_tiler_data,
)
from pipeline_utils.steps.processing.tiler_processing import (
    process,
)

context = create_picsellia_dataset_processing_context(
    processing_parameters_cls=ProcessingTilerParameters,
)


@pipeline(
    context=context,
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

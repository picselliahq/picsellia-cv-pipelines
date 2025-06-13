from albumentations_processing.steps import process
from albumentations_processing.utils.parameters import ProcessingParameters
from picsellia_cv_engine.core.services.utils.picsellia_context import (
    create_picsellia_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset

context = create_picsellia_processing_context(
    processing_parameters_cls=ProcessingParameters
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def albumentations_processing_pipeline():
    dataset_collection = load_coco_datasets()
    dataset_collection["output"] = process(
        dataset_collection["input"], dataset_collection["output"]
    )
    upload_full_dataset(dataset_collection["output"], use_id=False)
    return dataset_collection


if __name__ == "__main__":
    albumentations_processing_pipeline()

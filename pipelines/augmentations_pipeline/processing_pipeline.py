from examples.augmentation.utils.common import process_dataset
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.utils.picsellia_context import (
    create_picsellia_processing_context,
)
from picsellia_cv_engine.steps.data_extraction.processing_data_extractor import (
    get_processing_dataset_collection,
)
from picsellia_cv_engine.steps.data_upload.dataset_context_uploader import (
    upload_dataset_context,
)

processing_context = create_picsellia_processing_context(
    processing_parameters={
        "datalake": "default",
        "data_tag": "augmented",
        "num_augmentations": "3",
    }
)


@pipeline(
    context=processing_context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def augmentations_pipeline():
    dataset_collection = get_processing_dataset_collection()
    dataset_collection["output"] = process_dataset(
        dataset_collection["input"], dataset_collection["output"]
    )
    upload_dataset_context(dataset_collection["output"], use_id=False)
    return dataset_collection


if __name__ == "__main__":
    augmentations_pipeline()

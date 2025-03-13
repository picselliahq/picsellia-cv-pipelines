from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.utils.picsellia_context import (
    create_picsellia_processing_context,
)
from picsellia_cv_engine.steps.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.dataset.uploader import upload_full_dataset

from pipelines.augmentations_pipeline.process_dataset import process_dataset

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
    dataset_collection = load_coco_datasets()
    dataset_collection["output"] = process_dataset(
        dataset_collection["input"], dataset_collection["output"]
    )
    upload_full_dataset(dataset_collection["output"], use_id=False)
    return dataset_collection


if __name__ == "__main__":
    augmentations_pipeline()

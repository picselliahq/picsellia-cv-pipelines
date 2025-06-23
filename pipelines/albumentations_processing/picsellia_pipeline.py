from albumentations_processing.steps import process
from albumentations_processing.utils.parameters import ProcessingParameters
import os
from picsellia_cv_engine.core.services.utils.picsellia_context import (
    create_picsellia_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_full_dataset
from picsellia_cv_engine.core.services.data.dataset.uploader.utils import (
    configure_dataset_type,
    initialize_coco_data,
    upload_annotations,
)

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
    datalake_id = context.input_dataset_version.list_assets(limit=1)[0].get_data().datalake_id
    datalake = context.client.get_datalake(id=datalake_id)
    context.processing_parameters.datalake = datalake.name

    dataset = initialize_coco_data(dataset=dataset_collection["output"])
    annotations = dataset.coco_data.get("annotations", [])
    configure_dataset_type(dataset=dataset, annotations=annotations)
    data_tags: list[str] = [context.processing_parameters.data_tag]
    data = datalake.upload_data(
        filepaths=[
            os.path.join(dataset.images_dir, image_filename)
            for image_filename in os.listdir(dataset.images_dir)
        ],
        tags=data_tags,
    )
    job = dataset.dataset_version.add_data(data=data, wait=False)
    job.wait_for_done(blocking_time_increment = 5.0, attempts = 40)
    upload_annotations(dataset, False, True, True)
    return dataset_collection


if __name__ == "__main__":
    albumentations_processing_pipeline()

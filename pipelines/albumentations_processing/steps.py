from copy import deepcopy

from picsellia_cv_engine.core import CocoDataset, DatasetCollection
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.core.services.data.dataset.uploader.utils import (
    configure_dataset_type,
    initialize_coco_data,
    upload_images,
    upload_images_and_annotations,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from utils.parameters import ProcessingParameters
from utils.processing import process_images


@step
def process(dataset_collection: DatasetCollection[CocoDataset]) -> CocoDataset:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()
    parameters = context.processing_parameters
    asset_ids = context.asset_ids

    input_dataset = dataset_collection["input"]
    output_dataset = dataset_collection["output"]

    output_coco = deepcopy(input_dataset.coco_data)
    output_coco["images"] = []
    output_coco["annotations"] = []

    output_coco = process_images(
        input_images_dir=input_dataset.images_dir,
        input_coco=input_dataset.coco_data,
        parameters=parameters.to_dict(),
        output_images_dir=output_dataset.images_dir,
        output_coco=output_coco,
        inference_type=input_dataset.dataset_version.type,
        asset_ids=asset_ids,
    )
    output_dataset.coco_data = output_coco

    print("✅ Dataset processing complete!")
    return output_dataset


@step
def upload(dataset: CocoDataset) -> None:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()
    parameters = context.processing_parameters

    datalake = context.client.get_datalake(id=context.inputs.get("datalake"))
    data_tag = parameters.data_tag

    dataset = initialize_coco_data(dataset=dataset)
    annotations = dataset.coco_data.get("annotations", [])

    if annotations:
        configure_dataset_type(dataset=dataset, annotations=annotations)
        upload_images_and_annotations(
            dataset=dataset,
            datalake=datalake,
            data_tag=data_tag,
            use_id=False,
        )
    else:
        upload_images(dataset=dataset, datalake=datalake, data_tag=data_tag)

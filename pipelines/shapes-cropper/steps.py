from picsellia_cv_engine.core import (
    CocoDataset,
    DatasetCollection,
)
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_context import (
    PicselliaDatasetProcessingContext,
)
from picsellia_cv_engine.core.services.data.dataset.uploader.utils import (
    configure_dataset_type,
    initialize_coco_data,
    upload_images,
    upload_images_and_annotations,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.data_validator import ProcessingShapesCropperDataValidator
from utils.parameters import ProcessingParameters
from utils.processing import ShapesCropperProcessing


@step
def process(
    dataset_collection: DatasetCollection[CocoDataset],
) -> CocoDataset:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()

    processor = ShapesCropperProcessing(
        dataset_collection=dataset_collection,
        label_name_to_extract=context.inputs.get("label_name_to_extract"),
    )
    dataset_collection = processor.process()
    return dataset_collection["output"]


@step
def validate_shapes_cropper_data(
    dataset: CocoDataset,
) -> CocoDataset:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()

    validator = ProcessingShapesCropperDataValidator(
        dataset=dataset,
        client=context.client,
        label_name_to_extract=context.inputs.get("label_name_to_extract"),
        datalake_id=context.inputs.get("datalake"),
        fix_annotation=context.processing_parameters.fix_annotation,
    )
    dataset = validator.validate()
    return dataset


@step
def upload(dataset: CocoDataset) -> None:
    context: PicselliaDatasetProcessingContext[ProcessingParameters] = Pipeline.get_active_context()

    datalake = context.client.get_datalake(id=context.inputs.get("datalake"))
    data_tag = context.processing_parameters.data_tag

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

import argparse

from picsellia.exceptions import ResourceConflictError
from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core import CocoDataset, DatasetCollection
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets

from steps import process, upload, validate_tiler_data
from utils.parameters import (
    ProcessingTilerParameters,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.DATASET_VERSION_CREATION,
    processing_parameters_cls=ProcessingTilerParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def tiler_processing_pipeline() -> None:
    # _load_legacy_inputs looks for 'target_version_name' but the platform sends
    # the value under the input's declared name (e.g. 'Target_Dataset_Version_name').
    # Resolve the output dataset version here before load_coco_datasets() is called.
    if context.output_dataset_version.id == context.input_dataset_version.id:
        target_version_name = (
            context.inputs.get("Target_Dataset_Version_name")
            or context.inputs.get("target_version_name")
        )
        if not target_version_name:
            raise RuntimeError(
                f"Cannot resolve output dataset version. Current inputs: {context.inputs}"
            )
        dataset = context.client.get_dataset(name=context.input_dataset_version.name)
        try:
            context.output_dataset_version = dataset.create_version(version=target_version_name)
        except ResourceConflictError:
            context.output_dataset_version = dataset.get_version(version=target_version_name)

    dataset_collection = load_coco_datasets()
    dataset_collection["input"] = validate_tiler_data(
        dataset=dataset_collection["input"]
    )
    output_dataset = process(dataset_collection=dataset_collection)
    upload(dataset=output_dataset)


if __name__ == "__main__":
    tiler_processing_pipeline()

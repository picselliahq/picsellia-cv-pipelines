# type: ignore
import dataclasses
from argparse import ArgumentParser

from picsellia.types.enums import ProcessingType

from pipelines.bounding_box_cropper.pipeline_utils.steps.data_validation.processing_bounding_box_cropper_data_validator import (
    validate_bounding_box_cropper_data,
)
from pipelines.bounding_box_cropper.pipeline_utils.steps.processing.bounding_box_cropper_processing import (
    process,
)
from src.picsellia_cv_engine import pipeline
from src.picsellia_cv_engine.models.contexts.processing.local_picsellia_processing_context import (
    LocalPicselliaProcessingContext,
)
from src.picsellia_cv_engine.steps.data_extraction.processing_data_extractor import (
    get_processing_dataset_collection,
)
from src.picsellia_cv_engine.steps.data_upload.classification_dataset_context_uploader import (
    upload_classification_dataset_context,
)


@dataclasses.dataclass
class ProcessingBoundingBoxCropperParameters:
    label_name_to_extract: str = "person"
    datalake: str = "default"
    data_tag: str = None
    fix_annotation: bool = True


parser = ArgumentParser()
parser.add_argument("--api_token", type=str)
parser.add_argument("--organization_id", type=str)
parser.add_argument("--job_id", type=str)
parser.add_argument("--input_dataset_version_id", type=str)
parser.add_argument("--output_dataset_version_id", type=str)
parser.add_argument("--label_name_to_extract", type=str, default="person")
parser.add_argument("--datalake", type=str, default="default")
parser.add_argument("--data_tag", type=str)
parser.add_argument("--fix_annotation", action="store_true", default=False)

args = parser.parse_args()


def get_context() -> LocalPicselliaProcessingContext:
    return LocalPicselliaProcessingContext(
        api_token=args.api_token,
        organization_id=args.organization_id,
        job_id=args.job_id,
        job_type=ProcessingType.DATASET_VERSION_CREATION,
        download_annotations=False,
        input_dataset_version_id=args.input_dataset_version_id,
        output_dataset_version_id=args.output_dataset_version_id,
        processing_parameters=ProcessingBoundingBoxCropperParameters(
            label_name_to_extract=args.label_name_to_extract,
            datalake=args.datalake,
            data_tag=args.data_tag,
            fix_annotation=args.fix_annotation,
        ),
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def bounding_box_cropper_processing_pipeline() -> None:
    dataset_collection = get_processing_dataset_collection()
    validate_bounding_box_cropper_data(dataset_context=dataset_collection["input"])
    output_dataset_context = process(dataset_collection=dataset_collection)
    upload_classification_dataset_context(dataset_context=output_dataset_context)


if __name__ == "__main__":
    bounding_box_cropper_processing_pipeline()

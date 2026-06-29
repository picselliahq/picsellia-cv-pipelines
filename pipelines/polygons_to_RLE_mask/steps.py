import json
import os
from datetime import datetime

from picsellia import DatasetVersion
from picsellia.exceptions import ResourceConflictError
from picsellia.types.enums import ImportAnnotationMode, InferenceType
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import PicselliaDatasetProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from utils.processing import convert_annotations_to_rle


@step
def fork_dataset() -> DatasetVersion:
    """
    Fork `input_dataset_version` into a new MASK dataset version. Forking
    attaches the same underlying data to the new version server-side, so no
    file is downloaded or re-uploaded.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    input_dataset_version = context.input_dataset_version

    if input_dataset_version.type != InferenceType.SEGMENTATION:
        raise ValueError(
            "Input dataset version must be of type SEGMENTATION (polygons) to be "
            f"converted to RLE masks, got {input_dataset_version.type}."
        )

    version_name = context.inputs.get("output_dataset_version_name")
    if not version_name:
        raise ValueError("Input 'output_dataset_version_name' is required.")

    try:
        output_dataset_version, job = input_dataset_version.fork(
            version=version_name,
            type=InferenceType.MASK,
            with_tags=True,
            with_labels=False,
            with_annotations=False,
            wait=False,
        )
    except ResourceConflictError:
        version_name = f"{version_name}_{datetime.now().timestamp()}"
        print(
            f"A dataset version with that name already exists, forking as "
            f"'{version_name}' instead."
        )
        output_dataset_version, job = input_dataset_version.fork(
            version=version_name,
            type=InferenceType.MASK,
            with_tags=True,
            with_labels=False,
            with_annotations=False,
            wait=False,
        )
    job.wait_for_done(attempts=1000)

    for label in input_dataset_version.list_labels():
        output_dataset_version.create_label(label.name)

    print(
        f"Forked '{input_dataset_version.version}' into '{version_name}' "
        f"(type MASK) without re-uploading data, and recreated its labels."
    )
    return output_dataset_version


@step
def process(output_dataset_version: DatasetVersion) -> None:
    """
    Convert polygon segmentation annotations from `input_dataset_version` into
    COCO RLE-encoded masks and import them into `output_dataset_version`.
    """
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    input_dataset_version = context.input_dataset_version

    input_dataset = CocoDataset(name="input", dataset_version=input_dataset_version)
    input_dataset.download_annotations(
        destination_dir=os.path.join(context.working_dir, "annotations", "input"),
        use_id=False,
    )

    coco_data = convert_annotations_to_rle(input_dataset.coco_data)

    output_annotations_dir = os.path.join(
        context.working_dir, "annotations", "output"
    )
    os.makedirs(output_annotations_dir, exist_ok=True)
    coco_file_path = os.path.join(output_annotations_dir, "annotations.json")
    with open(coco_file_path, "w") as f:
        json.dump(coco_data, f)

    output_dataset_version.import_annotations_coco_file(
        file_path=coco_file_path,
        use_id=False,
        mode=ImportAnnotationMode.REPLACE,
    )

    print("Dataset processing complete!")

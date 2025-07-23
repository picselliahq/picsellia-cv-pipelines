from copy import deepcopy

from picsellia.types.enums import InferenceType
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.contexts import PicselliaProcessingContext
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from utils.processing import process_images

import logging
import os

from picsellia_cv_engine.core import CocoDataset, DatasetCollection, YoloDataset
from picsellia_cv_engine.core.contexts import (
    LocalProcessingContext,
    LocalTrainingContext,
    PicselliaProcessingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.services.data.dataset.loader import (
    TrainingDatasetCollectionExtractor,
)
from picsellia_cv_engine.core.services.data.dataset.validator.utils import (
    get_dataset_validator,
)
from picsellia_cv_engine.core.services.utils.dataset_logging import log_labelmap


@step
def process(input_dataset: CocoDataset, output_dataset: CocoDataset):
    """
    🚀 This function processes the dataset using `process_images()`.

    🔹 **What You Need to Do:**
    - Modify `process_images()` to apply custom transformations or augmentations.
    - Ensure it returns the correct processed images & COCO metadata.

    Args:
        input_dataset (CocoDataset): Input dataset from Picsellia.
        output_dataset (CocoDataset): Output dataset for saving processed data.

    Returns:
        CocoDataset: The processed dataset, ready for local execution and Picsellia.
    """

    # Get processing parameters from the user-defined configuration
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters.to_dict()
    client = context.client
    # Initialize an empty COCO dataset
    output_coco = deepcopy(input_dataset.coco_data)
    output_coco["images"] = []  # Reset image metadata
    output_coco["annotations"] = []  # Reset annotation metadata

    # Call the helper function to process images
    detection_dataset = client.get_dataset_version_by_id(
        parameters["detection_dataset_id"]
    )
    detection_dataset_coco = detection_dataset.build_coco_file_locally().model_dump()
    outptut_dataset_version = output_dataset.dataset_version
    labels = outptut_dataset_version.list_labels()
    for label in labels:
        label.delete()
    outptut_dataset_version.update(type=InferenceType.OBJECT_DETECTION)
    for label in detection_dataset.list_labels():
        outptut_dataset_version.create_label(label.name)
    output_coco = process_images(
        detection_dataset_coco=detection_dataset_coco,
        input_images_dir=input_dataset.images_dir,
        input_coco=input_dataset.coco_data,
        parameters=parameters,
        output_images_dir=output_dataset.images_dir,
        output_coco=output_coco,
    )
    # Assign processed data to output dataset
    output_dataset.coco_data = output_coco

    print(f"✅ Dataset processing complete!")
    return output_dataset


@step
def load_coco_datasets(
    skip_asset_listing: bool = False,
) -> DatasetCollection[CocoDataset] | CocoDataset:
    """
    A step for loading COCO datasets based on the current pipeline context (training or processing).

    This function adapts to different contexts and loads datasets accordingly:
    - **Training Contexts**: Loads datasets for training, validation, and testing splits.
    - **Processing Contexts**: Loads either a single dataset or multiple datasets depending on the context.

    Args:
        skip_asset_listing (bool, optional): Flag to determine whether to skip listing dataset assets before downloading.
            Default is `False`. This is applicable only for processing contexts.

    Returns:
        Union[DatasetCollection[CocoDataset], CocoDataset]: The loaded dataset(s) based on the context.

            - For **Training Contexts**: Returns a `DatasetCollection[CocoDataset]` containing training, validation,
              and test datasets.
            - For **Processing Contexts**:
                - If both input and output datasets are available, returns a `DatasetCollection[CocoDataset]`.
                - If only an input dataset is available, returns a single `CocoDataset` for the input dataset.

    Raises:
        ValueError:
            - If no datasets are found in the processing context.
            - If the context type is unsupported (neither training nor processing).

    Example:
        - In a **Training Context**, the function loads and prepares datasets for training, validation, and testing.
        - In a **Processing Context**, it loads the input and output datasets (if available) or just the input dataset.
    """
    context = Pipeline.get_active_context()
    return load_coco_datasets_impl(
        context=context, skip_asset_listing=skip_asset_listing
    )


def load_coco_datasets_impl(
    context: (
        PicselliaTrainingContext
        | LocalTrainingContext
        | PicselliaProcessingContext
        | LocalProcessingContext
    ),
    skip_asset_listing: bool,
) -> DatasetCollection[CocoDataset] | CocoDataset:
    """
    Implementation logic to load COCO datasets depending on the pipeline context type.

    Handles both training and processing contexts and downloads assets and annotations accordingly.

    Args:
        context: Either a training or processing context instance.
        skip_asset_listing (bool): Whether to skip asset listing before download.

    Returns:
        DatasetCollection[CocoDataset] or CocoDataset: The loaded dataset(s).

    Raises:
        ValueError: If no datasets are found or an unsupported context is provided.
    """
    # Training Context Handling
    if isinstance(context, PicselliaTrainingContext | LocalTrainingContext):
        dataset_collection_extractor = TrainingDatasetCollectionExtractor(
            experiment=context.experiment,
            train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
        )

        dataset_collection = dataset_collection_extractor.get_dataset_collection(
            context_class=CocoDataset,
            random_seed=context.hyperparameters.seed,
        )

        log_labelmap(
            labelmap=dataset_collection["train"].labelmap,
            experiment=context.experiment,
            log_name="labelmap",
        )

        dataset_collection.dataset_path = os.path.join(context.working_dir, "dataset")

        dataset_collection.download_all(
            images_destination_dir=os.path.join(
                dataset_collection.dataset_path, "images"
            ),
            annotations_destination_dir=os.path.join(
                dataset_collection.dataset_path, "annotations"
            ),
            use_id=True,
            skip_asset_listing=False,
        )

        return dataset_collection

    # Processing Context Handling
    elif isinstance(context, PicselliaProcessingContext | LocalProcessingContext):
        # If both input and output datasets are available
        if (
            context.input_dataset_version_id
            and context.output_dataset_version_id
            and not context.input_dataset_version_id
            == context.output_dataset_version_id
        ):
            input_dataset = CocoDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )
            output_dataset = CocoDataset(
                name="output",
                dataset_version=context.output_dataset_version,
                assets=None,
                labelmap=None,
            )
            dataset_collection = DatasetCollection([input_dataset, output_dataset])
            dataset_collection.download_all(
                images_destination_dir=os.path.join(context.working_dir, "images"),
                annotations_destination_dir=os.path.join(
                    context.working_dir, "annotations"
                ),
                use_id=False,
                skip_asset_listing=skip_asset_listing,
            )
            return dataset_collection

        # If only input dataset is available
        elif (
            context.input_dataset_version_id
            and context.input_dataset_version_id == context.output_dataset_version_id
        ) or (
            context.input_dataset_version_id and not context.output_dataset_version_id
        ):
            dataset = CocoDataset(
                name="input",
                dataset_version=context.input_dataset_version,
                assets=context.input_dataset_version.list_assets(),
                labelmap=None,
            )

            dataset.download_assets(
                destination_dir=os.path.join(
                    context.working_dir, "images", dataset.name
                ),
                use_id=True,
                skip_asset_listing=skip_asset_listing,
            )
            dataset.download_annotations(
                destination_dir=os.path.join(
                    context.working_dir, "annotations", dataset.name
                ),
                use_id=True,
            )

            return dataset

        else:
            raise ValueError("No datasets found in the processing context.")

    else:
        raise ValueError(f"Unsupported context type: {type(context)}")

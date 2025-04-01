import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection
from picsellia_cv_engine.services.base.data.dataset.preprocessing import (
    ClassificationBaseDatasetPreparator,
)


@step
def prepare_ultralytics_classification_dataset_collection(
    dataset_collection: DatasetCollection[CocoDataset],
) -> DatasetCollection[CocoDataset]:
    """
    Prepares and organizes a dataset collection for Ultralytics classification tasks.

    This function iterates over each dataset in the provided `DatasetCollection`, organizing them
    using the `ClassificationDatasetPreparator` to structure the dataset for use with Ultralytics classification.
    Each dataset is moved into a new directory, with the structure suitable for Ultralytics training.

    Args:
        dataset_collection (DatasetCollection): The original dataset collection to be prepared for classification.

    Returns:
        DatasetCollection: A dataset collection where each dataset has been organized and prepared for Ultralytics classification tasks.
    """
    context = Pipeline.get_active_context()
    for dataset in dataset_collection:
        destination_dir = str(
            os.path.join(
                os.getcwd(),
                context.experiment.name,
                "ultralytics_dataset",
                dataset.name,
            )
        )
        preparator = ClassificationBaseDatasetPreparator(
            dataset=dataset,
            destination_dir=destination_dir,
        )
        prepared_dataset = preparator.organize()

        dataset_collection[prepared_dataset.name] = prepared_dataset

    dataset_collection.dataset_path = os.path.join(
        os.getcwd(), context.experiment.name, "ultralytics_dataset"
    )

    return dataset_collection

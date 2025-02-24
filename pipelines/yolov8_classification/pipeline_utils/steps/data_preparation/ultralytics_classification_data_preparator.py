import os

from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.steps.data_preparation.classification_dataset_context_preparator import (
    ClassificationBaseDatasetContextPreparator,
)


@step
def prepare_ultralytics_classification_dataset_collection(
    dataset_collection: DatasetCollection[CocoDatasetContext],
) -> DatasetCollection[CocoDatasetContext]:
    """
    Prepares and organizes a dataset collection for Ultralytics classification tasks.

    This function iterates over each dataset context in the provided `DatasetCollection`, organizing them
    using the `ClassificationDatasetContextPreparator` to structure the dataset for use with Ultralytics classification.
    Each dataset is moved into a new directory, with the structure suitable for Ultralytics training.

    Args:
        dataset_collection (DatasetCollection): The original dataset collection to be prepared for classification.

    Returns:
        DatasetCollection: A dataset collection where each dataset has been organized and prepared for Ultralytics classification tasks.
    """
    context = Pipeline.get_active_context()
    for dataset_context in dataset_collection:
        destination_path = str(
            os.path.join(
                os.getcwd(),
                context.experiment.name,
                "ultralytics_dataset",
                dataset_context.dataset_name,
            )
        )
        preparator = ClassificationBaseDatasetContextPreparator(
            dataset_context=dataset_context,
            destination_path=destination_path,
        )
        prepared_dataset_context = preparator.organize()

        dataset_collection[prepared_dataset_context.dataset_name] = (
            prepared_dataset_context
        )

    dataset_collection.dataset_path = os.path.join(
        os.getcwd(), context.experiment.name, "ultralytics_dataset"
    )

    return dataset_collection

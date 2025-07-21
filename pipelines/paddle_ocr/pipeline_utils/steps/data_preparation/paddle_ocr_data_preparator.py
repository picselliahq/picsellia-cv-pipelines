import os

from picsellia_cv_engine.core import (
    CocoDataset,
    DatasetCollection,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from paddle_ocr.pipeline_utils.dataset.paddle_ocr_dataset_context import (
    PaddleOCRDataset,
)
from paddle_ocr.pipeline_utils.steps_utils.data_preparation.paddle_ocr_dataset_context_preparator import (
    PaddleOCRDatasetPreparator,
)


@step
def prepare_paddle_ocr_dataset_collection(
    dataset_collection: DatasetCollection[CocoDataset],
) -> DatasetCollection[PaddleOCRDataset]:
    """
    Prepares and organizes a dataset collection for PaddleOCR training.

    This function takes an existing `DatasetCollection` containing the 'train', 'val', and 'test' datasets,
    and organizes them into a format suitable for PaddleOCR training. It uses the `PaddleOCRDatasetPreparator`
    to organize the datasets (e.g., creating necessary directories and moving images) for each dataset split (train, val, test).
    The organized datasets are then stored in a new `DatasetCollection` with `PaddleOCRDataset` types.

    Args:
        dataset_collection (DatasetCollection[CocoDataset]): The original dataset collection containing 'train', 'val', and 'test' splits.

    Returns:
        DatasetCollection[PaddleOCRDataset]: A new dataset collection where each dataset is organized for PaddleOCR,
        with directories properly set up for training, validation, and testing.
    """
    context = Pipeline.get_active_context()

    paddleocr_dataset_collection = DatasetCollection(
        [
            PaddleOCRDatasetPreparator(
                dataset=dataset_collection["train"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["train"].name,
                    )
                ),
            ).organize(),
            PaddleOCRDatasetPreparator(
                dataset=dataset_collection["val"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["val"].name,
                    )
                ),
            ).organize(),
            PaddleOCRDatasetPreparator(
                dataset=dataset_collection["test"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["test"].name,
                    )
                ),
            ).organize(),
        ]
    )

    return paddleocr_dataset_collection

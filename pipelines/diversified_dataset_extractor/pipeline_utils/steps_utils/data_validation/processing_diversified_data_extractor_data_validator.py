from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.services.base.data.dataset.validator.common.dataset_validator import (
    DatasetValidator,
)


class ProcessingDiversifiedDataExtractorDataValidator(DatasetValidator):
    def __init__(
        self,
        dataset: TBaseDataset,
    ):
        super().__init__(dataset=dataset)

    def _validate_dataset_version_size(self) -> None:
        """
        Validate that the dataset version size is strictly greater than 1.

        Raises:
            ValueError: If the dataset version size is equal to 0 or 1.
        """
        dataset_version_size = self.dataset.dataset_version.sync()["size"]
        if dataset_version_size == 0:
            raise ValueError(
                "This dataset version cannot be diversified because it is empty. "
                "Please add some assets to the dataset version before running this processing."
            )

        elif dataset_version_size == 1:
            raise ValueError(
                "This dataset version has only one asset, therefore it cannot be diversified. "
                "Please add more assets to the dataset version before running this processing."
            )

    def validate(self) -> None:
        super().validate()
        self._validate_dataset_version_size()

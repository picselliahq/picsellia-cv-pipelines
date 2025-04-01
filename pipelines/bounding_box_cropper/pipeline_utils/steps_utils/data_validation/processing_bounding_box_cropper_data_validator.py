from picsellia import Client
from picsellia_cv_engine.core import (
    CocoDataset,
)
from picsellia_cv_engine.services.base.data.dataset.validator.object_detection.coco_object_detection_dataset_validator import (
    CocoObjectDetectionDatasetValidator,
)


class ProcessingBoundingBoxCropperDataValidator(CocoObjectDetectionDatasetValidator):
    def __init__(
        self,
        dataset: CocoDataset,
        client: Client,
        label_name_to_extract: str,
        datalake: str,
        fix_annotation: bool = True,
    ):
        super().__init__(dataset=dataset, fix_annotation=fix_annotation)
        self.client = client
        self.label_name_to_extract = label_name_to_extract
        self.datalake = datalake

    def _validate_label_name_to_extract(self) -> None:
        """
        Validate that the label name to extract is present in the labelmap.

        Raises:
            ValueError: If the label name to extract is not present in the labelmap.
        """
        if self.label_name_to_extract not in self.dataset.labelmap:
            raise ValueError(
                f"Label name {self.label_name_to_extract} is not present in the labelmap"
            )

    def _validate_datalake(self) -> None:
        """
        Validate that the datalake is valid.

        Raises:
            ValueError: If the datalake is not valid.
        """
        datalakes_name = [datalake.name for datalake in self.client.list_datalakes()]
        if self.datalake not in datalakes_name:
            raise ValueError(
                f"Datalake {self.datalake} is not valid, available datalakes are {datalakes_name}"
            )

    def validate(self) -> CocoDataset:
        self.dataset = super().validate()
        self._validate_label_name_to_extract()
        self._validate_datalake()
        return self.dataset

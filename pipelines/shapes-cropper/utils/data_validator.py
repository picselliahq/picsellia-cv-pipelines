from picsellia import Client
from picsellia.types.enums import InferenceType
from picsellia_cv_engine.core import CocoDataset
from picsellia_cv_engine.core.services.data.dataset.validator.object_detection.coco_object_detection_dataset_validator import (
    CocoObjectDetectionDatasetValidator,
)
from picsellia_cv_engine.core.services.data.dataset.validator.segmentation.coco_segmentation_dataset_validator import (
    CocoSegmentationDatasetValidator,
)


class ProcessingShapesCropperDataValidator(CocoObjectDetectionDatasetValidator):
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
        self.fix_annotation = fix_annotation

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
        """
        Run base COCO validations and shapes-cropper–specific checks.

        - For object-detection datasets: use `CocoObjectDetectionDatasetValidator`
          (via the superclass) then custom checks.
        - For segmentation datasets: run both object-detection and
          `CocoSegmentationDatasetValidator` (as segmentation COCO uses both
          bboxes and polygons), then custom checks.
        """
        # 1) Always run object-detection style validation first (superclass)
        self.dataset = super().validate()

        # 2) If it's a segmentation dataset, also run segmentation-specific validation
        if self.dataset.dataset_version.type == InferenceType.SEGMENTATION:
            segmentation_dataset_validator = CocoSegmentationDatasetValidator(
                dataset=self.dataset,
                fix_annotation=self.fix_annotation,
            )
            self.dataset = segmentation_dataset_validator.validate()

        # 3) Shapes-cropper–specific validation
        self._validate_label_name_to_extract()
        self._validate_datalake()
        return self.dataset

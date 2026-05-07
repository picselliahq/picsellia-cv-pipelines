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
        datalake_id: str,
        fix_annotation: bool = True,
    ):
        super().__init__(dataset=dataset, fix_annotation=fix_annotation)
        self.client = client
        self.label_name_to_extract = label_name_to_extract
        self.datalake_id = datalake_id
        self.fix_annotation = fix_annotation

    def _validate_label_name_to_extract(self) -> None:
        """
        Validate that all label names to extract are present in the labelmap.

        Raises:
            ValueError: If any label name to extract is not present in the labelmap.
        """
        labels = [l.strip() for l in self.label_name_to_extract.split(",") if l.strip()]
        if not labels:
            raise ValueError("'label_name_to_extract' must not be empty")
        missing = [label for label in labels if label not in self.dataset.labelmap]
        if missing:
            raise ValueError(
                f"Label(s) {missing} are not present in the labelmap"
            )

    def _validate_datalake(self) -> None:
        """
        Validate that the datalake ID is valid.

        Raises:
            ValueError: If the datalake ID is not valid.
        """
        try:
            self.client.get_datalake(id=self.datalake_id)
        except Exception:
            raise ValueError(f"Datalake '{self.datalake_id}' is not valid")

    def validate(self) -> CocoDataset:
        """
        Run base COCO validations and shapes-cropper–specific checks.

        - For object-detection datasets: use `CocoObjectDetectionDatasetValidator`
          (via the superclass) then custom checks.
        - For segmentation datasets: run both object-detection and
          `CocoSegmentationDatasetValidator` (as segmentation COCO uses both
          bboxes and polygons), then custom checks.
        """
        self.dataset = super().validate()

        if self.dataset.dataset_version.type == InferenceType.SEGMENTATION:
            segmentation_dataset_validator = CocoSegmentationDatasetValidator(
                dataset=self.dataset,
                fix_annotation=self.fix_annotation,
            )
            self.dataset = segmentation_dataset_validator.validate()

        self._validate_label_name_to_extract()
        self._validate_datalake()
        return self.dataset

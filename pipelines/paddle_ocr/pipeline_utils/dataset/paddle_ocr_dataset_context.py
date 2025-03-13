from picsellia import DatasetVersion, Label
from picsellia.sdk.asset import MultiAsset
from picsellia_cv_engine.models.data.dataset.coco_dataset_context import (
    CocoDatasetContext,
)


class PaddleOCRDatasetContext(CocoDatasetContext):
    """
    A specialized dataset context for handling PaddleOCR datasets.

    This class extends the generic DatasetContext to provide functionality specific to PaddleOCR,
    such as handling bounding box and text annotations as well as organizing the dataset structure
    for PaddleOCR tasks.

    Attributes:
        paddle_ocr_bbox_annotations_path (Optional[str]): Path to the bounding box annotations for OCR.
        paddle_ocr_text_annotations_path (Optional[str]): Path to the text annotations for OCR.
        text_images_dir (Optional[str]): Directory where the text-related images are stored.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_version: DatasetVersion,
        assets: MultiAsset,
        labelmap: dict[str, Label] | None = None,
    ):
        """
        Initializes the PaddleOCRDatasetContext with the specified dataset name, version, assets, and labelmap.

        Args:
            dataset_name (str): The name of the dataset.
            dataset_version (DatasetVersion): The version of the dataset in Picsellia.
            assets (MultiAsset): The assets associated with the dataset.
            labelmap (Optional[Dict[str, Label]]): Optional label map for the dataset.
        """
        super().__init__(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            assets=assets,
            labelmap=labelmap,
        )
        self.paddle_ocr_bbox_annotations_path: str | None = None
        self.paddle_ocr_text_annotations_path: str | None = None
        self.text_images_dir: str | None = None

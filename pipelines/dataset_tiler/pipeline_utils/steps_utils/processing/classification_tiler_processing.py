from typing import Any

from dataset_tiler.pipeline_utils.steps_utils.processing.base_tiler_processing import (
    BaseTilerProcessing,
)


class ClassificationTilerProcessing(BaseTilerProcessing):
    """Tiler processing for classification datasets."""

    def _adjust_coco_annotation(
        self,
        annotation: dict[str, Any],
        tile_x: int,
        tile_y: int,
        tile_info: dict[str, Any],
        output_coco_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Adjust annotation for a specific tile within a classification dataset.

        Args:
            annotation: Original annotation.
            tile_x: X-coordinate of the tile.
            tile_y: Y-coordinate of the tile.
            tile_info: Information about the tile in the output COCO data, usually containing the tile's id, width and height.
            output_coco_data: Output COCO data to which the adjusted annotation will be added.

        Returns:
            Optional[dict[str, Any]]: Adjusted annotation for the tile.
        """
        image_info = {
            "id": tile_info["id"],
            "file_name": tile_info["file_name"],
            "width": tile_info["width"],
            "height": tile_info["height"],
        }
        output_coco_data["images"].append(image_info)

        annotation = {
            "id": len(output_coco_data["annotations"]),
            "image_id": tile_info["id"],
            "category_id": annotation["category_id"],
            "bbox": [],
            "segmentation": [],
            "area": 0,
            "iscrowd": 0,
        }

        return annotation

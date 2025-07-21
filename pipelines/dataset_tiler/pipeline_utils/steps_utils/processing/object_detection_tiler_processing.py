from typing import Any

from dataset_tiler.pipeline_utils.steps_utils.processing.base_tiler_processing import (
    BaseTilerProcessing,
)


class ObjectDetectionTilerProcessing(BaseTilerProcessing):
    """Tiler processing for object detection datasets."""

    def _adjust_coco_annotation(
        self,
        annotation: dict[str, Any],
        tile_x: int,
        tile_y: int,
        tile_info: dict[str, Any],
        output_coco_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Adjust annotation for a specific tile.

        Args:
            annotation: Original COCO annotation.
            tile_x: X-coordinate of the tile.
            tile_y: Y-coordinate of the tile.
            tile_info: Information about the tile's image's COCO data. It contains the tile's id, filename, width and height.
            output_coco_data: Final COCO data that will be sent to Picsellia. Used to compute the new annotation id.

        Returns:
            Adjusted annotation or None if the annotation doesn't intersect with the tile.
        """
        bbox = annotation["bbox"]
        x, y, width, height = bbox

        # Calculate the intersection of the bounding box with the tile
        ix1 = max(x, tile_x)
        iy1 = max(y, tile_y)
        ix2 = min(x + width, tile_x + tile_info["width"])
        iy2 = min(y + height, tile_y + tile_info["height"])

        # Check if there's an intersection
        if ix2 <= ix1 or iy2 <= iy1:
            return None

        # Calculate new bounding box coordinates relative to the tile
        new_bbox_x = ix1 - tile_x
        new_bbox_y = iy1 - tile_y
        new_bbox_width = ix2 - ix1
        new_bbox_height = iy2 - iy1

        new_bbox = [new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height]
        new_bbox_area = new_bbox_width * new_bbox_height

        # Check if the intersection area is large enough
        if not self._should_bounding_box_annotation_be_kept(bbox, new_bbox):
            return None

        # Create the new annotation
        new_annotation = annotation.copy()
        new_annotation["id"] = len(output_coco_data["annotations"])
        new_annotation["image_id"] = tile_info["id"]
        new_annotation["bbox"] = new_bbox
        new_annotation["area"] = new_bbox_area

        return new_annotation

    def _should_bounding_box_annotation_be_kept(
        self, old_bounding_box: list[int], new_bounding_box: list[int]
    ) -> bool:
        """
        Check if the annotation should be kept based on the new bounding box.

        Args:
            old_bounding_box: Original bounding box.
            new_bounding_box: New bounding box.

        Returns:
            True if the annotation should be kept, False otherwise.
        """
        old_x, old_y, old_width, old_height = old_bounding_box
        new_x, new_y, new_width, new_height = new_bounding_box

        # Check if the new bounding box is large enough
        if new_width < old_width:
            if (
                self.min_annotation_width is not None
                and new_width < self.min_annotation_width
            ):
                return False

        # Check if the new bounding box is tall enough
        if new_height < old_height:
            if (
                self.min_annotation_height is not None
                and new_height < self.min_annotation_height
            ):
                return False

        # Check the area as well
        old_area = old_width * old_height
        new_area = new_width * new_height

        if (
            self.min_annotation_area_ratio is not None
            and new_area / old_area < self.min_annotation_area_ratio
        ):
            return False

        return True

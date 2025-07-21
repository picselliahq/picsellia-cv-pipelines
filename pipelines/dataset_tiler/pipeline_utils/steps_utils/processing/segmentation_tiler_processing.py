from typing import Any

from dataset_tiler.pipeline_utils.steps_utils.processing.object_detection_tiler_processing import (
    ObjectDetectionTilerProcessing,
)


class SegmentationTilerProcessing(ObjectDetectionTilerProcessing):
    """Tiler processing for segmentation datasets."""

    def _adjust_coco_annotation(
        self,
        annotation: dict[str, Any],
        tile_x: int,
        tile_y: int,
        tile_info: dict[str, Any],
        output_coco_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Adjust annotation for a specific tile within a segmentation dataset.

        Args:
            annotation: Original COCO annotation.
            tile_x: X-coordinate of the tile.
            tile_y: Y-coordinate of the tile.
            tile_info: Information about the tile's image's COCO data. It contains the tile's id, filename, width and height.
            output_coco_data: Final COCO data that will be sent to Picsellia. Used to compute the new annotation id.

        Returns:
            Optional[dict[str, Any]]: Adjusted annotation for the tile.
        """
        # Adjust the bounding box first
        new_annotation = super()._adjust_coco_annotation(
            annotation=annotation,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_info=tile_info,
            output_coco_data=output_coco_data,
        )

        if new_annotation is None:
            return None

        adjusted_segmentation = self._adjust_segmentation_annotation(
            polygons=annotation["segmentation"],
            tile_x=tile_x,
            tile_y=tile_y,
            tile_width=tile_info["width"],
            tile_height=tile_info["height"],
        )

        if adjusted_segmentation is None or len(adjusted_segmentation) == 0:
            return None

        new_annotation["segmentation"] = adjusted_segmentation

        return new_annotation

    def _clip_polygon(self, polygon, clip_rect):
        """Clip a polygon to a rectangle using Sutherland-Hodgman algorithm."""

        def inside(p, cp1, cp2):
            return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (
                p[0] - cp1[0]
            )

        def compute_intersection(cp1, cp2, s, e):
            dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
            dp = [s[0] - e[0], s[1] - e[1]]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

        output_list = polygon
        cp1 = clip_rect[-1]

        for clip_vertex in clip_rect:
            cp2 = clip_vertex
            input_list = output_list
            output_list = []

            if not input_list:
                break

            s = input_list[-1]

            for subject_vertex in input_list:
                e = subject_vertex
                if inside(e, cp1, cp2):
                    if not inside(s, cp1, cp2):
                        output_list.append(compute_intersection(cp1, cp2, s, e))
                    output_list.append(e)
                elif inside(s, cp1, cp2):
                    output_list.append(compute_intersection(cp1, cp2, s, e))
                s = e
            cp1 = cp2

        return output_list

    def _adjust_segmentation_annotation(
        self,
        polygons: list[list[float]],
        tile_x: int,
        tile_y: int,
        tile_width: int,
        tile_height: int,
    ) -> list[list[float]]:
        """
        Adjust segmentation coordinates for a tile and clip the polygon to the tile boundaries.

        Args:
            polygons (List[List[float]]): Original segmentation coordinates in COCO format.
            tile_x (int): X-coordinate of the tile's top-left corner.
            tile_y (int): Y-coordinate of the tile's top-left corner.
            tile_width (int): Width of the tile.
            tile_height (int): Height of the tile.

        Returns:
            List[List[float]]: Adjusted and clipped segmentation coordinates.
        """

        adjusted_segmentation = []
        clip_rect = [
            (0, 0),
            (tile_width, 0),
            (tile_width, tile_height),
            (0, tile_height),
        ]

        for polygon in polygons:
            # Convert flat list to list of points
            points = [
                (polygon[i] - tile_x, polygon[i + 1] - tile_y)
                for i in range(0, len(polygon), 2)
            ]

            # Clip the polygon
            clipped_poly = self._clip_polygon(points, clip_rect)

            # Convert back to flat list and add to adjusted segmentation
            if (
                len(clipped_poly) > 2
            ):  # Only add if the polygon is valid (at least 3 points)
                adjusted_poly = [
                    abs(coord) for point in clipped_poly for coord in point
                ]
                adjusted_segmentation.append(adjusted_poly)

        return adjusted_segmentation

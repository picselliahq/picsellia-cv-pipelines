from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
from picsellia.sdk.asset import Asset
from PIL import Image

from evaluator.utils.general import (
    cast_type_list_to_float,
    cast_type_list_to_int,
    convert_tensor_to_list,
    rescale_normalized_box,
)


class FrameworkFormatter(ABC):
    def __init__(self, labelmap) -> None:
        self._labelmap = labelmap

    @abstractmethod
    def format_classes(self, prediction) -> List[int]:
        pass

    @abstractmethod
    def format_confidences(self, prediction) -> List[float]:
        pass

    @abstractmethod
    def format_boxes(self, asset: Asset, prediction) -> List[int]:
        pass

    @abstractmethod
    def format_polygons(self, asset: Asset, prediction):
        pass


class YoloFormatter(FrameworkFormatter):
    def format_confidences(self, prediction):
        if prediction.boxes is not None:
            confidences_list = convert_tensor_to_list(tensor=prediction.boxes.conf)

        elif prediction.probs is not None:
            confidences_list = [max(prediction.probs)]
        else:
            return []
        casted_confidences = cast_type_list_to_float(_list=confidences_list)
        return casted_confidences

    def format_classes(self, prediction):
        if prediction.boxes is not None:
            classes_list = convert_tensor_to_list(tensor=prediction.boxes.cls)
        elif prediction.probs is not None:
            confidences_list = convert_tensor_to_list(tensor=prediction.probs)
            classes_list = [confidences_list.index(max(confidences_list))]
        else:
            return []
        casted_classes = cast_type_list_to_int(_list=classes_list)
        picsellia_labels = list(
            map(lambda label: self._labelmap[label], casted_classes)
        )
        return picsellia_labels

    def format_boxes(self, asset: Asset, prediction):
        if not prediction.boxes:
            return []
        normalized_boxes = prediction.boxes.xyxyn
        boxes_list = convert_tensor_to_list(tensor=normalized_boxes)
        rescaled_boxes = list(
            map(
                lambda box: rescale_normalized_box(
                    box=box, width=asset.width, height=asset.height
                ),
                boxes_list,
            )
        )
        casted_boxes = list(map(cast_type_list_to_int, rescaled_boxes))
        return casted_boxes

    def format_polygons(self, asset: Asset, prediction):
        if prediction.masks is None:
            return []
        polygons = prediction.masks.xy
        casted_polygons = list(map(lambda polygon: polygon.astype(int), polygons))
        return list(map(lambda polygon: polygon.tolist(), casted_polygons))


class TensorflowFormatter(FrameworkFormatter):
    def format_confidences(self, prediction):
        if prediction["detection_scores"] is not None:
            scores = prediction["detection_scores"].numpy()[0].astype(np.float).tolist()

        return scores

    def format_classes(self, prediction):
        if prediction["detection_scores"] is not None:
            classes = (
                prediction["detection_classes"].numpy()[0].astype(np.float).tolist()
            )
        casted_classes = cast_type_list_to_int(_list=classes)
        picsellia_labels = list(
            map(lambda label: self._labelmap[label], casted_classes)
        )
        return picsellia_labels

        return classes

    def format_boxes(self, asset: Asset, prediction):
        boxes = self._postprocess_boxes(
            prediction["detection_boxes"].numpy()[0].astype(np.float).tolist(),
            asset.width,
            asset.height,
        )

        return boxes

    def format_polygons(self, asset: Asset, prediction):
        boxes = self._postprocess_boxes(
            prediction["detection_boxes"].numpy()[0].astype(np.float).tolist(),
            asset.width,
            asset.height,
        )
        masks = self._postprocess_masks(
            detection_masks=prediction["detection_masks"]
            .numpy()[0]
            .astype(np.float)
            .tolist(),
            resized_detection_boxes=boxes,
            mask_threshold=0.4,
            image_height=asset.height,
            image_width=asset.width,
        )

        return masks

    def _postprocess_boxes(
        self, detection_boxes: list, image_width: int, image_height: int
    ) -> list:
        return [
            [
                int(e[1] * image_width),
                int(e[0] * image_height),
                int((e[3] - e[1]) * image_width),
                int((e[2] - e[0]) * image_height),
            ]
            for e in detection_boxes
        ]

    def _postprocess_masks(
        self,
        detection_masks: list,
        resized_detection_boxes: list,
        image_width: int,
        image_height: int,
        mask_threshold: float = 0.5,
    ) -> list:
        list_mask = []
        for idx, detection_mask in enumerate(detection_masks):
            # background_mask with all black=0
            mask = np.zeros((image_height, image_width))
            # Get normalised bbox coordinates
            xmin, ymin, w, h = resized_detection_boxes[idx]
            if w > 0 and h > 0 and xmin > 0 and ymin > 0:
                xmax = xmin + w
                ymax = ymin + h

                # Define bbox height and width
                bbox_height, bbox_width = h, w

                # Resize 'detection_mask' to bbox size
                bbox_mask = np.array(
                    Image.fromarray(np.array(detection_mask) * 255).resize(
                        size=(bbox_width, bbox_height), resample=Image.BILINEAR
                    )
                    # Image.NEAREST is fastest and no weird artefacts
                )
                # Insert detection_mask into image.size np.zeros((height, width)) background_mask
                assert bbox_mask.shape == mask[ymin:ymax, xmin:xmax].shape
                mask[ymin:ymax, xmin:xmax] = bbox_mask
                if (
                    mask_threshold > 0
                ):  # np.where(mask != 1, 0, mask)  # in case threshold is used to have other values (0)
                    mask = np.where(np.abs(mask) > mask_threshold * 255, 1, mask)
                    mask = np.where(mask != 1, 0, mask)

                try:
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_TC89_KCOS,
                    )
                    to_add = (
                        contours[len(contours) - 1][::1]
                        .reshape(
                            contours[len(contours) - 1][::1].shape[0],
                            contours[len(contours) - 1][::1].shape[2],
                        )
                        .tolist()
                    )
                    list_mask.append(to_add)
                except Exception:
                    pass  # No contours

        return list_mask


class KerasClassificationFormatter(FrameworkFormatter):
    def format_confidences(self, prediction):
        pass

    def format_classes(self, prediction):
        pass

    def format_boxes(self, asset: Asset, prediction):
        pass

    def format_polygons(self, prediction):
        pass

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from PIL import Image
import cv2
from picsellia.types.enums import InferenceType
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class AbstractFormatter(ABC):
    @abstractmethod
    def format_output(self, raw_output, model_type):
        pass

    @abstractmethod
    def format_object_detection(self, raw_output):
        pass

    @abstractmethod
    def format_segmentation(self, raw_output):
        pass

    # @abstractmethod
    # def format_classification(self, raw_output):
    #     pass


class TensorflowFormatter(AbstractFormatter):
    def __init__(self, image_width, image_height, output_names):
        self.image_height = image_height
        self.image_width = image_width
        self.output_names = output_names

    def format_output(self, raw_output: dict, model_type: InferenceType):
        if model_type == InferenceType.OBJECT_DETECTION:
            return self.format_object_detection(raw_output)
        elif model_type == InferenceType.SEGMENTATION:
            return self.format_segmentation(raw_output)
        # elif model_type == InferenceType.CLASSIFICATION:
        #     return self.format_classification(raw_output)

    def format_object_detection(self, raw_output):
        try:
            scores = raw_output["detection_scores"].numpy()[0].astype(np.float).tolist()
            boxes = self._postprocess_boxes(
                raw_output["detection_boxes"].numpy()[0].astype(np.float).tolist()
            )
            classes = (
                raw_output["detection_classes"].numpy()[0].astype(np.float).tolist()
            )
        except KeyError:
            boxes, scores, classes = self._guess_output_names(raw_output=raw_output)
            boxes = self._postprocess_boxes(boxes)
        response = {
            "detection_scores": scores,
            "detection_boxes": boxes,
            "detection_classes": classes,
        }
        return response

    def _guess_output_names(self, raw_output) -> Tuple[list, list, list]:
        boxes, scores, classes = [], [], []
        possible_choices = ["bbox", "classes", "scores", "num_detections"]
        for output_name in self.output_names:
            unknown_layer = raw_output[output_name]
            if len(unknown_layer.shape) == 3:
                assert "bbox" in possible_choices
                boxes = unknown_layer[0].astype(np.float).tolist()
                possible_choices.remove("bbox")
            elif len(unknown_layer.shape) == 1:
                assert "num_detections" in possible_choices
                possible_choices.remove("num_detections")
            elif unknown_layer.dtype == np.float32:
                assert "scores" in possible_choices
                scores = unknown_layer[0].astype(np.float).tolist()
                possible_choices.remove("scores")
            else:
                assert "classes" in possible_choices
                classes = unknown_layer[0].astype(np.int16).tolist()
                possible_choices.remove("classes")
        assert len(possible_choices) == 0
        return (boxes, scores, classes)

    def format_segmentation(self, raw_output):
        scores = (
            raw_output["detection_scores"].numpy()[0].astype(np.float).tolist()[:10]
        )
        boxes = self._postprocess_boxes(
            raw_output["detection_boxes"].numpy()[0].astype(np.float).tolist()
        )
        masks = self._postprocess_masks(
            detection_masks=raw_output["detection_masks"]
            .numpy()[0]
            .astype(np.float)
            .tolist()[:10],
            resized_detection_boxes=boxes,
            mask_threshold=0.4,
        )
        classes = (
            raw_output["detection_classes"].numpy()[0].astype(np.float).tolist()[:10]
        )
        response = {
            "detection_scores": scores,
            "detection_boxes": boxes,
            "detection_masks": masks,
            "detection_classes": classes,
        }

        return response

    def format_classification(self, raw_output):
        output_name = self.output_names[0]
        scores = ([float(max(raw_output[output_name].numpy()[0]))],)
        classes = ([int(np.argmax(raw_output[output_name].numpy()[0]))],)

        return (scores, classes)

    def _postprocess_boxes(self, detection_boxes: list) -> list:
        return [
            [
                int(e[1] * self.image_width),
                int(e[0] * self.image_height),
                int((e[3] - e[1]) * self.image_width),
                int((e[2] - e[0]) * self.image_height),
            ]
            for e in detection_boxes
        ]

    def _postprocess_masks(
        self,
        detection_masks: list,
        resized_detection_boxes: list,
        mask_threshold: float = 0.5,
    ) -> list:
        list_mask = []
        for idx, detection_mask in enumerate(detection_masks):
            # background_mask with all black=0
            mask = np.zeros((self.image_height, self.image_width))
            print(resized_detection_boxes[idx])
            # Get normalised bbox coordinates
            xmin, ymin, w, h = resized_detection_boxes[idx]

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

import logging
from abc import abstractmethod
from typing import List

import numpy as np
from picsellia.exceptions import PicselliaError
from picsellia.sdk.asset import Asset
from PIL import UnidentifiedImageError
from ultralytics import YOLO

from evaluator.abstract_evaluator import AbstractEvaluator
from evaluator.framework_formatter import YoloFormatter
from evaluator.type_formatter import (
    ClassificationFormatter,
    DetectionFormatter,
    SegmentationFormatter,
    TypeFormatter,
)
from evaluator.utils.general import open_asset_as_array


class YOLOEvaluator(AbstractEvaluator):
    framework_formatter = YoloFormatter

    @abstractmethod
    def _get_model_artifact_filename(self):
        pass

    def _load_saved_model(self):
        try:
            self._loaded_model = YOLO(
                self._model_weights_path, task=self._get_model_task()
            )
            logging.info("Model loaded in memory.")
        except Exception as e:
            raise PicselliaError(
                f"Impossible to load saved model located at: {self._model_weights_path}"
            ) from e

    @abstractmethod
    def _get_model_task(self):
        pass

    def _preprocess_images(self, assets: List[Asset]) -> List[np.array]:
        images = []
        for asset in assets:
            try:
                image_data = open_asset_as_array(asset)

                # Since YOLO from ultralytics except numpy arrays as BGR, we need to convert the image to BGR
                image_data = image_data[:, :, ::-1]

            except UnidentifiedImageError:
                logging.warning(
                    f"Can't evaluate {asset.filename}, error opening the image"
                )
                continue

            images.append(image_data)
        return images

    def _get_model_weights_path(self):
        pass


class ClassificationYOLOEvaluator(YOLOEvaluator):
    type_formatter: type[TypeFormatter] = ClassificationFormatter

    def _get_model_artifact_filename(self):
        return "weights"

    def _get_model_task(self):
        return "classify"


class DetectionYOLOEvaluator(YOLOEvaluator):
    type_formatter: type[TypeFormatter] = DetectionFormatter

    def _get_model_artifact_filename(self):
        return "checkpoint-index-latest"

    def _get_model_task(self):
        return "detect"


class SegmentationYOLOEvaluator(YOLOEvaluator):
    type_formatter: type[TypeFormatter] = SegmentationFormatter

    def _get_model_artifact_filename(self):
        return "checkpoint-index-latest"

    def _get_model_task(self):
        return "segment"

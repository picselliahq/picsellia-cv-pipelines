from abc import ABC, abstractmethod
from typing import Any

from picsellia.sdk.asset import Asset

from evaluator.framework_formatter import FrameworkFormatter


class TypeFormatter(ABC):
    def __init__(self, framework_formatter: FrameworkFormatter) -> None:
        self._framework_formatter = framework_formatter

    @abstractmethod
    def format_prediction(self, asset: Asset, prediction):
        pass

    @abstractmethod
    def format_evaluation(self, picsellia_prediction):
        pass

    @abstractmethod
    def get_shape_type(self):
        pass


class ClassificationFormatter(TypeFormatter):
    def format_prediction(
        self, asset: Asset, prediction
    ) -> dict[str, list[float] | list[int]]:
        return {
            "confidences": self._framework_formatter.format_confidences(
                prediction=prediction
            ),
            "classes": self._framework_formatter.format_classes(prediction=prediction),
        }

    def format_evaluation(self, picsellia_prediction):
        return picsellia_prediction["classes"], picsellia_prediction["confidences"]

    def get_shape_type(self):
        return "classifications"


class DetectionFormatter(TypeFormatter):
    def format_prediction(
        self, asset: Asset, prediction
    ) -> dict[str, list[int] | list[float]]:
        return {
            "boxes": self._framework_formatter.format_boxes(
                asset=asset, prediction=prediction
            ),
            "confidences": self._framework_formatter.format_confidences(
                prediction=prediction
            ),
            "classes": self._framework_formatter.format_classes(prediction=prediction),
        }

    def format_evaluation(self, picsellia_prediction):
        return (
            *picsellia_prediction["boxes"],
            picsellia_prediction["classes"],
            picsellia_prediction["confidences"],
        )

    def get_shape_type(self):
        return "rectangles"


class SegmentationFormatter(TypeFormatter):
    def format_prediction(
        self, asset: Asset, prediction
    ) -> dict[str, list[float] | list[int] | Any]:
        return {
            "polygons": self._framework_formatter.format_polygons(
                prediction=prediction
            ),
            "confidences": self._framework_formatter.format_confidences(
                prediction=prediction
            ),
            "classes": self._framework_formatter.format_classes(prediction=prediction),
        }

    def format_evaluation(self, picsellia_prediction):
        return (
            picsellia_prediction["polygons"],
            picsellia_prediction["classes"],
            picsellia_prediction["confidences"],
        )

    def get_shape_type(self):
        return "polygons"

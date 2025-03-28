from picsellia import Experiment
from picsellia_cv_engine.models.steps.model.logging import (
    BaseLogger,
    Metric,
)

from pipelines.yolov8.training.pipeline_utils.steps_utils.model_logging.ultralytics_base_model_logger import (
    UltralyticsBaseMetricMapping,
)


class UltralyticsObjectDetectionMetricMapping(UltralyticsBaseMetricMapping):
    """ """

    def __init__(self):
        """ """
        super().__init__()
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="box_loss", framework_name="train/box_loss"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="cls_loss", framework_name="train/cls_loss"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="dfl_loss", framework_name="train/dfl_loss"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="labels", framework_name="labels"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(
                standard_name="labels_correlogram", framework_name="labels_correlogram"
            ),
        )

        self.add_metric(
            phase="val",
            metric=Metric(standard_name="box_loss", framework_name="val/box_loss"),
        )
        self.add_metric(
            phase="val",
            metric=Metric(standard_name="cls_loss", framework_name="val/cls_loss"),
        )
        self.add_metric(
            phase="val",
            metric=Metric(standard_name="dfl_loss", framework_name="val/dfl_loss"),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="precision(B)", framework_name="metrics/precision(B)"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="recall(B)", framework_name="metrics/recall(B)"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(standard_name="mAP50(B)", framework_name="metrics/mAP50(B)"),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="mAP50-95(B)", framework_name="metrics/mAP50-95(B)"
            ),
        )


class UltralyticsObjectDetectionLogger(BaseLogger):
    """ """

    def __init__(
        self,
        experiment: Experiment,
        metric_mapping: UltralyticsObjectDetectionMetricMapping,
    ):
        """ """
        super().__init__(experiment=experiment, metric_mapping=metric_mapping)

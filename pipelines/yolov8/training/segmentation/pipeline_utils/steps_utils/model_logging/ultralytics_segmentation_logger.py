from picsellia import Experiment
from picsellia_cv_engine.models.steps.model.logging.base_logger import (
    BaseLogger,
    Metric,
)
from picsellia_cv_engine.models.steps.model.logging.classification_logger import (
    ClassificationMetricMapping,
)


class UltralyticsSegmentationMetricMapping(ClassificationMetricMapping):
    """ """

    def __init__(self):
        """ """
        super().__init__()
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="seg_loss", framework_name="train/seg_loss"),
        )
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
            metric=Metric(standard_name="learning_rate", framework_name="lr/pg0"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="learning_rate_pg1", framework_name="lr/pg1"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="learning_rate_pg2", framework_name="lr/pg2"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="epoch_time", framework_name="epoch_time"),
        )

        self.add_metric(
            phase="val",
            metric=Metric(standard_name="seg_loss", framework_name="val/seg_loss"),
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
                standard_name="precision(M)", framework_name="metrics/precision(M)"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="recall(M)", framework_name="metrics/recall(M)"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(standard_name="mAP50(M)", framework_name="metrics/mAP50(M)"),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="mAP50-95(M)", framework_name="metrics/mAP50-95(M)"
            ),
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
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="precision(M)-final",
                framework_name="metrics/precision(M)-final",
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="recall(M)-final",
                framework_name="metrics/recall(M)-final",
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="mAP50(M)-final", framework_name="metrics/mAP50(M)-final"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="mAP50-95(M)-final",
                framework_name="metrics/mAP50-95(M)-final",
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="precision(B)-final",
                framework_name="metrics/precision(B)-final",
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="recall(B)-final",
                framework_name="metrics/recall(B)-final",
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="mAP50(B)-final", framework_name="metrics/mAP50(B)-final"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="mAP50-95(B)-final",
                framework_name="metrics/mAP50-95(B)-final",
            ),
        )


class UltralyticsSegmentationLogger(BaseLogger):
    """ """

    def __init__(
        self,
        experiment: Experiment,
        metric_mapping: UltralyticsSegmentationMetricMapping,
    ):
        """ """
        super().__init__(experiment=experiment, metric_mapping=metric_mapping)

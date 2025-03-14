from picsellia import Experiment
from picsellia_cv_engine.models.steps.model.logging.base_logger import (
    BaseLogger,
    Metric,
)
from picsellia_cv_engine.models.steps.model.logging.classification_logger import (
    ClassificationMetricMapping,
)


class UltralyticsClassificationMetricMapping(ClassificationMetricMapping):
    """
    Defines the metric mapping for classification tasks in the Ultralytics framework.

    This class extends the ClassificationMetricMapping and adds specific framework-related metric names
    for training and validation phases, such as top-1 and top-5 accuracy, loss, and learning rate.
    """

    def __init__(self):
        """
        Initializes the Ultralytics-specific classification metric mapping.
        """
        super().__init__()
        self.add_metric(
            phase="train",
            metric=Metric(
                standard_name="accuracy", framework_name="metrics/accuracy_top1"
            ),
        )
        self.add_metric(
            phase="train",
            metric=Metric(
                standard_name="accuracy_top5", framework_name="metrics/accuracy_top5"
            ),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="loss", framework_name="train/loss"),
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
            metric=Metric(
                standard_name="accuracy", framework_name="metrics/accuracy_top1"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="accuracy_top5", framework_name="metrics/accuracy_top5"
            ),
        )
        self.add_metric(
            phase="val", metric=Metric(standard_name="loss", framework_name="val/loss")
        )


class UltralyticsClassificationLogger(BaseLogger):
    """
    Logger for Ultralytics-based classification models.

    This class logs classification metrics during training and validation phases in Ultralytics models,
    using a metric mapping specific to the Ultralytics framework.
    """

    def __init__(self, experiment: Experiment):
        """
        Initialize the UltralyticsClassificationLogger with an experiment and Ultralytics metric mapping.

        Args:
            experiment (Experiment): The experiment object for logging Ultralytics classification metrics.
        """
        metric_mapping = UltralyticsClassificationMetricMapping()
        super().__init__(experiment=experiment, metric_mapping=metric_mapping)

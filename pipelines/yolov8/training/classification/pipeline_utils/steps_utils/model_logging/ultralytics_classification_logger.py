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
            phase="train",
            metric=Metric(standard_name="batch_0", framework_name="train_batch0"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="batch_1", framework_name="train_batch1"),
        )
        self.add_metric(
            phase="train",
            metric=Metric(standard_name="batch_2", framework_name="train_batch2"),
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
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="batch_0_labels", framework_name="val_batch0_labels"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="batch_1_labels", framework_name="val_batch1_labels"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="batch_2_labels", framework_name="val_batch2_labels"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="batch_0_preds", framework_name="val_batch0_pred"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="batch_1_preds", framework_name="val_batch1_pred"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="batch_2_preds", framework_name="val_batch2_pred"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="confusion_matrix", framework_name="confusion_matrix"
            ),
        )
        self.add_metric(
            phase="val",
            metric=Metric(
                standard_name="confusion_matrix_normalized",
                framework_name="confusion_matrix_normalized",
            ),
        )


class UltralyticsClassificationLogger(BaseLogger):
    """
    Logger for Ultralytics-based classification models.

    This class logs classification metrics during training and validation phases in Ultralytics models,
    using a metric mapping specific to the Ultralytics framework.
    """

    def __init__(
        self, experiment: Experiment, metric_mapping: ClassificationMetricMapping
    ):
        """
        Initialize the UltralyticsClassificationLogger with an experiment and Ultralytics metric mapping.

        Args:
            experiment (Experiment): The experiment object for logging Ultralytics classification metrics.
        """
        super().__init__(experiment=experiment, metric_mapping=metric_mapping)

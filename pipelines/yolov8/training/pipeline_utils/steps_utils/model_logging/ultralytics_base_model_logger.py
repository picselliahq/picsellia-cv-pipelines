from picsellia_cv_engine.models.steps.model.logging.base_logger import (
    Metric,
    MetricMapping,
)


class UltralyticsBaseMetricMapping(MetricMapping):
    def __init__(self):
        super().__init__()
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

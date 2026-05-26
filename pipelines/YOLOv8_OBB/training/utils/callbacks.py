from pathlib import Path

from picsellia_cv_engine.frameworks.ultralytics.services.model.callbacks import (
    TBaseValidator,
    UltralyticsCallbacks,
)


class UltralyticsObbCallbacks(UltralyticsCallbacks):
    """OBB-aware callbacks. The OBBValidator does not expose the per-class
    instance/image counts that the base implementation logs, so we only emit
    the per-class metric values returned by `validator.metrics.class_result`."""

    def on_val_end(self, validator: TBaseValidator):
        val_output_directory = Path(validator.save_dir)

        valid_prefixes = ("val", "P", "R", "F1", "Box", "Mask")
        for file_path in val_output_directory.iterdir():
            if file_path.stem.startswith(valid_prefixes):
                self.logger.log_image(
                    name=file_path.stem, image_path=str(file_path), phase="val"
                )

        if not (
            hasattr(validator, "metrics")
            and hasattr(validator.metrics, "ap_class_index")
        ):
            return

        table_data = []
        row_labels = []
        for i, c in enumerate(validator.metrics.ap_class_index):
            row_labels.append(validator.names[c])
            metrics = validator.metrics.class_result(i)
            table_data.append([round(float(m), 3) for m in metrics])

        if not table_data:
            return

        columns = ["P", "R", "mAP50", "mAP50-95"][: len(table_data[0])]

        self.logger.log_table(
            name="metrics",
            data={
                "data": table_data,
                "rows": row_labels,
                "columns": columns,
            },
            phase="val",
        )

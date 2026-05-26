from picsellia_cv_engine.frameworks.ultralytics.parameters.hyper_parameters import (
    UltralyticsHyperParameters,
)


class TrainingHyperParameters(UltralyticsHyperParameters):
    """YOLOv8-OBB training hyperparameters.

    Inherits the full Ultralytics hyperparameter surface (optimizer, schedule,
    loss gains, regularization, etc.) plus the base HyperParameters
    (epochs, batch_size, image_size, seed, validate, train_set_split_ratio,
    device)."""

from picsellia_cv_engine import pipeline
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.parameters import ExportParameters
from picsellia_cv_engine.frameworks.ultralytics.parameters.augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from picsellia_cv_engine.frameworks.ultralytics.parameters.hyper_parameters import (
    UltralyticsHyperParameters,
)
from picsellia_cv_engine.frameworks.ultralytics.steps.dataset.preparator import (
    prepare_ultralytics_dataset,
)
from picsellia_cv_engine.frameworks.ultralytics.steps.model.evaluator import (
    evaluate_ultralytics_model,
)
from picsellia_cv_engine.frameworks.ultralytics.steps.model.exporter import (
    export_ultralytics_model,
)
from picsellia_cv_engine.frameworks.ultralytics.steps.model.loader import (
    load_ultralytics_model,
)

from pipelines.yolov8.training.utils.training_steps import (
    simple_train_ultralytics_model,
)


def get_context() -> PicselliaTrainingContext[
    UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
]:
    return PicselliaTrainingContext(
        hyperparameters_cls=UltralyticsHyperParameters,
        augmentation_parameters_cls=UltralyticsAugmentationParameters,
        export_parameters_cls=ExportParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_training_pipeline():
    dataset_collection = prepare_ultralytics_dataset()

    model = load_ultralytics_model(pretrained_weights_name="pretrained-weights")

    simple_train_ultralytics_model(model=model, dataset_collection=dataset_collection)

    export_ultralytics_model(model=model)

    evaluate_ultralytics_model(model=model, dataset=dataset_collection["test"])


if __name__ == "__main__":
    yolov8_training_pipeline()

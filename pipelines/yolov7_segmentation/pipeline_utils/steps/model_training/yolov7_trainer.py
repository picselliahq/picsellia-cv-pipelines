from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.models.parameters.export_parameters import (
    ExportParameters,
)

from pipelines.yolov7_segmentation.pipeline_utils.dataset.yolov7_dataset_collection import (
    Yolov7DatasetCollection,
)
from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model import (
    Yolov7Model,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_augmentation_parameters import (
    Yolov7AugmentationParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)
from pipelines.yolov7_segmentation.pipeline_utils.steps_utils.model_training.yolov7_model_trainer import (
    Yolov7ModelTrainer,
)


@step
def yolov7_model_trainer(
    model: Yolov7Model, dataset_collection: Yolov7DatasetCollection
) -> Yolov7Model:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_trainer = Yolov7ModelTrainer(model=model, experiment=context.experiment)

    if (
        not context.api_token
        or not context.organization_id
        or not context.experiment_id
        or not context.host
    ):
        raise ValueError(
            "API token, organization ID, experiment ID, and host must be set"
        )

    model_trainer.train_model(
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
        api_token=context.api_token,
        organization_id=context.organization_id,
        host=context.host,
        experiment_id=context.experiment_id,
    )

    model.set_trained_weights_path()

    return model

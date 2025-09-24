import os
import os.path

from picsellia_cv_engine.core import (
    YoloDataset,
)
from picsellia_cv_engine.core.contexts import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.data import (
    TBaseDataset,
)
from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.core.services.data.dataset.loader import (
    TrainingDatasetCollectionExtractor,
)
from picsellia_cv_engine.core.services.utils.dataset_logging import log_labelmap
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel
from picsellia_cv_engine.steps.base.model.evaluator import evaluate_model_impl
from utils.dataset import Yolov7DatasetCollection
from utils.model import Yolov7Model
from utils.model_prediction import Yolov7SegmentationModelPredictor
from utils.model_training import Yolov7ModelTrainer
from utils.parameters import Yolov7AugmentationParameters, Yolov7HyperParameters


@step
def get_dataset_collection() -> Yolov7DatasetCollection:
    """
    Extracts datasets from an experiment and prepares them for training, validation, and testing.

    This function retrieves the active training context from the pipeline and uses it to initialize a
    `TrainingDatasetCollectionExtractor` with the current experiment and the training split ratio from the
    hyperparameters. It retrieves a `DatasetCollection` of datasets ready for use in training, validation,
    and testing, downloading all necessary assets and annotations.

    The function also logs the labelmap and the objects distribution for each dataset split in the collection,
    facilitating data analysis and tracking in the experiment.

    Returns:
        DatasetCollection: A collection of datasets prepared for training, validation, and testing,
        with all necessary assets and annotations downloaded.

    Raises:
        ResourceNotFoundError: If any of the expected dataset splits (train, validation, test) are not found in the experiment.
        RuntimeError: If an invalid number of datasets are attached to the experiment.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    dataset_collection_extractor = TrainingDatasetCollectionExtractor(
        experiment=context.experiment,
        train_set_split_ratio=context.hyperparameters.train_set_split_ratio,
    )

    yolo_dataset_collection = dataset_collection_extractor.get_dataset_collection(
        context_class=YoloDataset,
        random_seed=context.hyperparameters.seed,
    )

    log_labelmap(
        labelmap=yolo_dataset_collection["train"].labelmap,
        experiment=context.experiment,
        log_name="labelmap",
    )

    yolov7_dataset_collection = Yolov7DatasetCollection(
        datasets=list(yolo_dataset_collection.datasets.values())
    )

    yolov7_dataset_collection.dataset_path = os.path.join(
        context.working_dir, "dataset"
    )

    yolov7_dataset_collection.download_all(
        images_destination_dir=os.path.join(
            yolov7_dataset_collection.dataset_path, "images"
        ),
        annotations_destination_dir=os.path.join(
            yolov7_dataset_collection.dataset_path, "labels"
        ),
        use_id=True,
        skip_asset_listing=True,
    )

    return yolov7_dataset_collection


@step
def prepare_dataset_collection(
    dataset_collection: Yolov7DatasetCollection,
) -> Yolov7DatasetCollection:
    if not dataset_collection.dataset_path:
        raise ValueError("Dataset path is not set in the dataset collection.")
    dataset_collection.write_config(
        config_path=os.path.join(dataset_collection.dataset_path, "dataset_config.yaml")
    )
    return dataset_collection


@step
def get_model(
    pretrained_weights_name: str | None = None,
    trained_weights_name: str | None = None,
    config_name: str | None = None,
    hyperparameters_name: str | None = None,
    exported_weights_name: str | None = None,
) -> Yolov7Model:
    """
    Extracts a model from the active Picsellia training experiment.

    This function retrieves the active training context from the pipeline and extracts the base model version
    from the experiment. It then creates a `Model` object for the model, specifying the name and pretrained
    weights. The function downloads the necessary model weights to a specified directory and returns the
    initialized `Model`.

    Returns:
        Model: The extracted and initialized model with the downloaded weights.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    model_version = context.experiment.get_base_model_version()
    model = Yolov7Model(
        name=model_version.name,
        model_version=model_version,
        pretrained_weights_name=pretrained_weights_name,
        trained_weights_name=trained_weights_name,
        config_name=config_name,
        hyperparameters_name=hyperparameters_name,
        exported_weights_name=exported_weights_name,
    )
    model.download_weights(destination_dir=os.path.join(context.working_dir, "model"))
    model.set_hyperparameters_path(
        destination_path=os.path.join(context.working_dir, "model", "weights")
    )
    return model


@step
def load_model(model: Yolov7Model, weights_path_to_load: str) -> Yolov7Model:
    context: PicselliaTrainingContext = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        loaded_model = UltralyticsModel.load_yolo_weights(
            weights_path=weights_path_to_load,
            device=context.hyperparameters.device,
        )
        model.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model


@step
def prepare_model(
    model: Yolov7Model,
) -> Yolov7Model:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    input_hyperparameters = {
        "lr0": context.hyperparameters.lr0,
        "lrf": context.hyperparameters.lrf,
        "momentum": context.hyperparameters.momentum,
        "weight_decay": context.hyperparameters.weight_decay,
        "warmup_epochs": context.hyperparameters.warmup_epochs,
        "warmup_momentum": context.hyperparameters.warmup_momentum,
        "warmup_bias_lr": context.hyperparameters.warmup_bias_lr,
        "box": context.hyperparameters.box,
        "cls": context.hyperparameters.cls,
        "cls_pw": context.hyperparameters.cls_pw,
        "obj": context.hyperparameters.obj,
        "obj_pw": context.hyperparameters.obj_pw,
        "iou_t": context.hyperparameters.iou_t,
        "anchor_t": context.hyperparameters.anchor_t,
        "fl_gamma": context.hyperparameters.fl_gamma,
        "hsv_h": context.augmentation_parameters.hsv_h,
        "hsv_s": context.augmentation_parameters.hsv_s,
        "hsv_v": context.augmentation_parameters.hsv_v,
        "degrees": context.augmentation_parameters.degrees,
        "translate": context.augmentation_parameters.translate,
        "scale": context.augmentation_parameters.scale,
        "shear": context.augmentation_parameters.shear,
        "perspective": context.augmentation_parameters.perspective,
        "flipud": context.augmentation_parameters.flipud,
        "fliplr": context.augmentation_parameters.fliplr,
        "mosaic": context.augmentation_parameters.mosaic,
        "mixup": context.augmentation_parameters.mixup,
        "copy_paste": context.augmentation_parameters.copy_paste,
        "paste_in": context.augmentation_parameters.paste_in,
        "loss_ota": context.hyperparameters.loss_ota,
    }

    if not model.hyperparameters_path:
        raise (ValueError("Hyperparameters path is not set"))

    model.update_hyperparameters(
        hyperparameters=input_hyperparameters,
        hyperparameters_path=model.hyperparameters_path,
    )

    return model


@step
def train_model(
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


@step
def evaluate_yolov7_model(
    model: Yolov7Model,
    dataset: TBaseDataset,
) -> None:
    context: PicselliaTrainingContext[
        Yolov7HyperParameters, Yolov7AugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_predictor = Yolov7SegmentationModelPredictor(model=model)
    image_paths = model_predictor.pre_process_dataset(dataset=dataset)
    label_path_to_mask_paths = model_predictor.run_inference(
        image_paths=image_paths,
        hyperparameters=context.hyperparameters,
    )
    picsellia_polygons_predictions = model_predictor.post_process(
        label_path_to_mask_paths=label_path_to_mask_paths,
        dataset=dataset,
    )

    evaluate_model_impl(
        context=context,
        picsellia_predictions=picsellia_polygons_predictions,
        inference_type=model.model_version.type,
        assets=dataset.assets,
        output_dir=os.path.join(model.results_dir, "inference"),
        training_labelmap=dict(enumerate(model.labelmap.keys())),
    )

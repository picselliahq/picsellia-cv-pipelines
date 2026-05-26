import os

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel
from picsellia_cv_engine.frameworks.ultralytics.services.model.logger.object_detection import (
    UltralyticsObjectDetectionLogger,
    UltralyticsObjectDetectionMetricMapping,
)

from utils.callbacks import UltralyticsObbCallbacks
from utils.data import prepare_obb_dataset
from utils.predictor import UltralyticsObbModelPredictor


@step()
def train(
    picsellia_model: UltralyticsModel,
    picsellia_datasets: DatasetCollection[CocoDataset],
):
    context = Pipeline.get_active_context()
    hyperparameters = context.hyperparameters
    augmentation_parameters = context.augmentation_parameters

    data_yaml_path = prepare_obb_dataset(picsellia_datasets=picsellia_datasets)

    yolo_model = picsellia_model.loaded_model

    callback_handler = UltralyticsObbCallbacks(
        experiment=context.experiment,
        logger=UltralyticsObjectDetectionLogger,
        metric_mapping=UltralyticsObjectDetectionMetricMapping(),
        model=picsellia_model,
        save_period=hyperparameters.save_period,
    )
    for event, fn in callback_handler.get_callbacks().items():
        yolo_model.add_callback(event, fn)

    yolo_model.train(
        # Core
        data=data_yaml_path,
        epochs=hyperparameters.epochs,
        time=hyperparameters.time,
        patience=hyperparameters.patience,
        batch=hyperparameters.batch_size,
        imgsz=hyperparameters.image_size,
        save=True,
        save_period=hyperparameters.save_period,
        cache=hyperparameters.cache,
        device=hyperparameters.device,
        workers=hyperparameters.workers,
        project=picsellia_model.results_dir,
        name=picsellia_model.name,
        exist_ok=True,
        pretrained=True,
        # Optimizer / schedule
        optimizer=hyperparameters.optimizer,
        seed=hyperparameters.seed,
        deterministic=hyperparameters.deterministic,
        single_cls=hyperparameters.single_cls,
        rect=hyperparameters.rect,
        cos_lr=hyperparameters.cos_lr,
        close_mosaic=hyperparameters.close_mosaic,
        amp=hyperparameters.amp,
        fraction=hyperparameters.fraction,
        profile=hyperparameters.profile,
        freeze=hyperparameters.freeze,
        lr0=hyperparameters.lr0,
        lrf=hyperparameters.lrf,
        momentum=hyperparameters.momentum,
        weight_decay=hyperparameters.weight_decay,
        warmup_epochs=hyperparameters.warmup_epochs,
        warmup_momentum=hyperparameters.warmup_momentum,
        warmup_bias_lr=hyperparameters.warmup_bias_lr,
        # Loss gains
        box=hyperparameters.box,
        cls=hyperparameters.cls,
        dfl=hyperparameters.dfl,
        label_smoothing=hyperparameters.label_smoothing,
        nbs=hyperparameters.nbs,
        dropout=hyperparameters.dropout,
        val=hyperparameters.validate,
        plots=hyperparameters.plots,
        # Augmentations
        hsv_h=augmentation_parameters.hsv_h,
        hsv_s=augmentation_parameters.hsv_s,
        hsv_v=augmentation_parameters.hsv_v,
        degrees=augmentation_parameters.degrees,
        translate=augmentation_parameters.translate,
        scale=augmentation_parameters.scale,
        shear=augmentation_parameters.shear,
        perspective=augmentation_parameters.perspective,
        flipud=augmentation_parameters.flipud,
        fliplr=augmentation_parameters.fliplr,
        bgr=augmentation_parameters.bgr,
        mosaic=augmentation_parameters.mosaic,
        mixup=augmentation_parameters.mixup,
        copy_paste=augmentation_parameters.copy_paste,
        auto_augment=augmentation_parameters.auto_augment,
        erasing=augmentation_parameters.erasing,
        crop_fraction=augmentation_parameters.crop_fraction,
    )

    picsellia_model.set_latest_run_dir()
    picsellia_model.set_trained_weights_path()
    if not picsellia_model.trained_weights_path:
        raise FileNotFoundError("Trained weights path could not be resolved.")
    picsellia_model.save_artifact_to_experiment(
        artifact_name="best-model",
        artifact_path=picsellia_model.trained_weights_path,
    )

    _evaluate_on_test_split(
        picsellia_model=picsellia_model,
        picsellia_datasets=picsellia_datasets,
        context=context,
    )


def _evaluate_on_test_split(picsellia_model, picsellia_datasets, context):
    test_dataset = picsellia_datasets.datasets.get("test")
    if test_dataset is None or not test_dataset.assets:
        print("ℹ️ No test split available — skipping final evaluation upload.")
        return

    predictor = UltralyticsObbModelPredictor(model=picsellia_model)

    image_paths = predictor.pre_process_dataset(dataset=test_dataset)
    image_batches = predictor.prepare_batches(
        image_paths=image_paths, batch_size=context.hyperparameters.batch_size
    )
    batch_results = predictor.run_inference_on_batches(image_batches=image_batches)
    picsellia_predictions = predictor.post_process_batches(
        image_batches=image_batches,
        batch_results=batch_results,
        dataset=test_dataset,
    )

    evaluate_model_impl(
        context=context,
        picsellia_predictions=picsellia_predictions,
        # OBB predictions are uploaded as 4-point polygons; SEGMENTATION is the
        # matching Picsellia inference type for polygon evaluations.
        inference_type=picsellia_model.model_version.type,
        assets=test_dataset.assets,
        output_dir=os.path.join(context.working_dir, "evaluation"),
        training_labelmap=context.experiment.get_log("labelmap").data,
    )

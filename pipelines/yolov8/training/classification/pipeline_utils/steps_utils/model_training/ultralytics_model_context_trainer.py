import os

from picsellia import Experiment
from picsellia_cv_engine.models.data.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.data.dataset.dataset_collection import (
    DatasetCollection,
)

from pipelines.yolov8.training.classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)
from pipelines.yolov8.training.classification.pipeline_utils.parameters.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from pipelines.yolov8.training.classification.pipeline_utils.parameters.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps_utils.model_logging.ultralytics_classification_logger import (
    UltralyticsClassificationLogger,
    UltralyticsClassificationMetricMapping,
)
from pipelines.yolov8.training.classification.pipeline_utils.steps_utils.model_training.ultralytics_callbacks import (
    UltralyticsCallbacks,
)
from pipelines.yolov8.training.object_detection.pipeline_utils.steps_utils.model_logging.ultralytics_object_detection_logger import (
    UltralyticsObjectDetectionLogger,
    UltralyticsObjectDetectionMetricMapping,
)


class UltralyticsModelContextTrainer:
    """
    Trainer class for handling the training process of a model using the Ultralytics framework.
    """

    def __init__(
        self,
        model_context: UltralyticsModelContext,
        experiment: Experiment,
    ):
        """
        Initializes the trainer with a model context and an experiment.

        Args:
            model_context (ModelContext): The context of the model to be trained.
            experiment (Experiment): The experiment instance used for logging and tracking.
        """
        self.model_context = model_context
        self.experiment = experiment
        if self.model_context.loaded_model.task == "classif":
            self.callback_handler = UltralyticsCallbacks(
                experiment,
                logger=UltralyticsClassificationLogger,
                metric_mapping=UltralyticsClassificationMetricMapping(),
            )
        elif self.model_context.loaded_model.task == "detect":
            self.callback_handler = UltralyticsCallbacks(
                experiment,
                logger=UltralyticsObjectDetectionLogger,
                metric_mapping=UltralyticsObjectDetectionMetricMapping(),
            )

    def _setup_callbacks(self):
        """
        Sets up the callbacks for the model training process.
        """
        for event, callback in self.callback_handler.get_callbacks().items():
            self.model_context.loaded_model.add_callback(event, callback)

    def train_model_context(
        self,
        dataset_collection: DatasetCollection[TBaseDatasetContext],
        hyperparameters: UltralyticsHyperParameters,
        augmentation_parameters: UltralyticsAugmentationParameters,
    ) -> UltralyticsModelContext:
        """
        Trains the model within the provided context using the given datasets, hyperparameters, and augmentation parameters.

        Args:
            dataset_collection (DatasetCollection): The collection of datasets used for training.
            hyperparameters (UltralyticsHyperParameters): The hyperparameters used for training.
            augmentation_parameters (UltralyticsAugmentationParameters): The augmentation parameters applied during training.

        Returns:
            ModelContext: The updated model context after training.
        """

        self._setup_callbacks()

        if self.model_context.loaded_model.task == "classif":
            data = dataset_collection.dataset_path
        else:
            data = os.path.join(dataset_collection.dataset_path, "data.yaml")

        if hyperparameters.epochs > 0:
            self.model_context.loaded_model.train(
                # Hyperparameters
                data=data,
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
                project=self.model_context.results_dir,
                name=self.model_context.model_name,
                exist_ok=True,
                pretrained=True,
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
                box=hyperparameters.box,
                cls=hyperparameters.cls,
                dfl=hyperparameters.dfl,
                pose=hyperparameters.pose,
                kobj=hyperparameters.kobj,
                label_smoothing=hyperparameters.label_smoothing,
                nbs=hyperparameters.nbs,
                overlap_mask=hyperparameters.overlap_mask,
                mask_ratio=hyperparameters.mask_ratio,
                dropout=hyperparameters.dropout,
                val=hyperparameters.validate,
                plots=hyperparameters.plots,
                # Augmentation parameters
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

        return self.model_context

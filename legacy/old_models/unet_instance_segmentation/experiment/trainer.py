import logging
import os
import sys

import keras
import matplotlib.image
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from picsellia.exceptions import ResourceNotFoundError
from picsellia.sdk.dataset import DatasetVersion
from picsellia.types.enums import AnnotationFileType, InferenceType
from pycocotools.coco import COCO
from skimage.measure import approximate_polygon, find_contours

sys.path.append(os.path.join(os.getcwd(), "unet-instance-segmentation", "experiment"))

from mask_to_polygon_converter.custom_converter import CustomConverter
from utils import (
    Dataloader,
    Dataset,
    convert_mask_to_binary,
    find_asset_by_dataset_index,
    format_and_log_eval_metrics,
    format_polygons,
    get_classes_mask_dataset,
    get_classes_segmentation_dataset,
    get_filename_from_fullpath,
    get_image_annotations,
    get_mask_from_annotations,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation,
    log_training_sample_to_picsellia,
    makedirs_images_masks,
    move_files_for_polygon_creation,
    move_images_and_masks_to_directories,
    predict_and_log_mask,
    predict_mask_from_image,
    shift_x_and_y_coordinates,
    split_train_test_val_filenames,
)

from abstract_trainer.trainer import AbstractTrainer


class UnetSegmentationTrainer(AbstractTrainer):
    def __init__(self):
        super().__init__()

        self.image_dataset = None
        self.mask_dataset = None
        self.segmentation_dataset = None
        self.n_classes = None
        self.classes = None
        self.image_dataset_name = "images"
        self.label = None
        self.eval_dataset_version = None
        self.annotated_dataset = None
        self.annotation_file_path = None
        self.mask_files = None
        self.image_files = None
        self.image_path = os.path.join(self.experiment.png_dir, "original")
        self.mask_path = os.path.join(self.experiment.png_dir, "masks")

        self.x_train_dir = os.path.join(self.experiment.png_dir, "train-images")
        self.y_train_dir = os.path.join(self.experiment.png_dir, "train-masks")

        self.x_test_dir = os.path.join(self.experiment.png_dir, "test-images")
        self.y_test_dir = os.path.join(self.experiment.png_dir, "test-masks")

        self.x_eval_dir = os.path.join(self.experiment.png_dir, "eval-images")
        self.y_eval_dir = os.path.join(self.experiment.png_dir, "eval-masks")

        self.backbone = "efficientnetb1"
        self.parameters = self.experiment.get_log("parameters").data
        self.mask_prefix = str(self.parameters.get("mask_filename_prefix", ""))
        self.image_prefix = str(self.parameters.get("image_filename_prefix", ""))

        self.preprocess_input = sm.get_preprocessing(self.backbone)

        self.batch_size = int(self.parameters.get("batch_size", 4))
        self.best_model_path = os.path.join(
            self.experiment.checkpoint_dir, "best_model.h5"
        )
        self.saved_model_path = os.path.join(self.experiment.base_dir, "model_latest")
        self.metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
        ]

        self.train_dataloader = None
        self.test_dataloader = None
        self.eval_dataloader = None
        self.model = None
        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.best_model_path,
                save_weights_only=True,
                save_best_only=True,
                mode="min",
            ),
            keras.callbacks.ReduceLROnPlateau(),
        ]
        self.conversion_tolerance = 0.9
        self.min_contour_points = 20

    def prepare_data_for_training(self):
        self.segmentation_dataset = self.get_segmentation_dataset()
        if self.segmentation_dataset is not None:
            self.download_segmentation_data_into_masks()
            self.classes = get_classes_segmentation_dataset(
                segmentation_dataset=self.segmentation_dataset
            )
        else:
            self.mask_dataset, self.image_dataset = self.get_mask_image_datasets()
            self._download_mask_image_datasets()
            self._get_mask_image_filenames()
            self.classes = get_classes_mask_dataset(mask_dataset=self.mask_dataset)

        self._split_and_move_data()
        self._create_train_test_eval_dataloaders()

    def get_segmentation_dataset(self) -> DatasetVersion | None:
        try:
            segmentation_dataset = self.experiment.get_dataset(name="full")
            self.image_dataset_name = "full"
        except ResourceNotFoundError:
            segmentation_dataset = None
        return segmentation_dataset

    def download_segmentation_data_into_masks(self):
        self._download_segmentation_dataset()
        self.export_annotations_from_segmentation_dataset()
        self.convert_all_images_polygons_to_masks()

    def get_mask_image_datasets(
        self,
    ) -> tuple[DatasetVersion, DatasetVersion]:
        try:
            mask_dataset = self.experiment.get_dataset(name="masks")
            image_dataset = self.experiment.get_dataset(name="images")
            return mask_dataset, image_dataset
        except ResourceNotFoundError:
            raise Exception(
                "You need to have either 'full' containing the annotated images (segmentation dataset), or 'masks' and 'images' to train with 'masks'"
            )

    def _download_segmentation_dataset(self):
        self.segmentation_dataset.download(target_path=self.image_path)
        self.image_files = os.listdir(path=self.image_path)

    def export_annotations_from_segmentation_dataset(self):
        self.annotation_file_path = self.segmentation_dataset.export_annotation_file(
            target_path=self.experiment.base_dir,
            annotation_file_type=AnnotationFileType.COCO,
        )

    def convert_all_images_polygons_to_masks(self):
        coco = COCO(self.annotation_file_path)
        os.makedirs(self.mask_path)
        for image_id in coco.getImgIds():
            self.convert_image_polygons_to_mask(coco=coco, image_id=image_id)
        self.mask_files = os.listdir(self.mask_path)

    def convert_image_polygons_to_mask(self, coco: COCO, image_id: int):
        img = coco.imgs[image_id]
        img_filename_without_extension = os.path.splitext(img["file_name"])[0]
        image_annotations = get_image_annotations(coco, img)
        mask = get_mask_from_annotations(coco=coco, image_annotations=image_annotations)
        binary_mask = convert_mask_to_binary(mask)
        self.save_binary_mask(filename=img_filename_without_extension, mask=binary_mask)

    def save_binary_mask(self, filename: str, mask: np.ndarray):
        filename = filename + ".png"
        matplotlib.image.imsave(
            os.path.join(self.mask_path, filename),
            mask,
            cmap="Greys",
            format="png",
        )

    def _download_mask_image_datasets(self):
        self.mask_dataset.download(target_path=self.mask_path)
        self.image_dataset.download(target_path=self.image_path)

    def _get_mask_image_filenames(self):
        self.image_files = os.listdir(path=self.image_path)
        self.mask_files = os.listdir(path=self.mask_path)

    def _split_and_move_data(self):
        (
            self.train_images_filenames,
            self.test_images_filenames,
            self.eval_images_filenames,
        ) = split_train_test_val_filenames(image_files=self.image_files, seed=11)
        makedirs_images_masks(
            x_train_dir=self.x_train_dir,
            y_train_dir=self.y_train_dir,
            x_test_dir=self.x_test_dir,
            y_test_dir=self.y_test_dir,
            x_eval_dir=self.x_eval_dir,
            y_eval_dir=self.y_eval_dir,
        )
        self._move_all_images_masks_to_directories()

    def _move_all_images_masks_to_directories(self):
        dataset_mappings = [
            (self.train_images_filenames, self.x_train_dir, self.y_train_dir),
            (self.test_images_filenames, self.x_test_dir, self.y_test_dir),
            (self.eval_images_filenames, self.x_eval_dir, self.y_eval_dir),
        ]

        for image_list, dest_image_dir, dest_mask_dir in dataset_mappings:
            move_images_and_masks_to_directories(
                image_path=self.image_path,
                mask_path=self.mask_path,
                image_list=image_list,
                mask_list=self.mask_files,
                dest_image_dir=dest_image_dir,
                dest_mask_dir=dest_mask_dir,
                mask_prefix=self.mask_prefix,
                image_prefix=self.image_prefix,
            )

    def _create_train_test_eval_dataloaders(self):
        train_dataset = Dataset(
            self.x_train_dir,
            self.y_train_dir,
            classes=self.classes,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(self.preprocess_input),
        )
        test_dataset = Dataset(
            self.x_test_dir,
            self.y_test_dir,
            classes=self.classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocess_input),
        )
        self.eval_dataset = Dataset(
            self.x_eval_dir,
            self.y_eval_dir,
            classes=self.classes,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocess_input),
        )

        log_training_sample_to_picsellia(
            dataset=train_dataset,
            experiment=self.experiment,
        )
        self.train_dataloader = Dataloader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)
        self.eval_dataloader = Dataloader(
            self.eval_dataset, batch_size=1, shuffle=False
        )

    def train(self):
        epochs = int(self.parameters.get("epochs", 1))
        learning_rate = self.parameters.get("learning_rate", 5e-4)
        self.n_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        activation = "sigmoid" if self.n_classes == 1 else "softmax"

        (
            loaded_checkpoint_path,
            loaded_finetuned_checkpoint_path,
        ) = self._download_either_original_or_finetuned_artifact()

        if loaded_checkpoint_path and not loaded_finetuned_checkpoint_path:
            self.model = sm.Unet(
                self.backbone,
                classes=self.n_classes,
                activation=activation,
                encoder_weights=loaded_checkpoint_path,
            )
        elif loaded_finetuned_checkpoint_path and not loaded_checkpoint_path:
            self.model = sm.Unet(
                self.backbone, classes=self.n_classes, activation=activation
            )
            self.model.load_weights(loaded_finetuned_checkpoint_path)

        optim = keras.optimizers.Adam(learning_rate)
        total_loss = self._get_total_loss()
        self.model.compile(optim, total_loss, self.metrics)

        _ = self.model.fit(
            self.train_dataloader,
            steps_per_epoch=len(self.train_dataloader),
            epochs=epochs,
            callbacks=self.callbacks,
            validation_data=self.eval_dataloader,
            validation_steps=len(self.eval_dataloader),
        )
        self.experiment.store("finetuned_model_weights", self.best_model_path)

        tf.saved_model.save(
            self.model, os.path.join(self.experiment.exported_model_dir, "saved_model")
        )
        self.experiment.store(
            "model-latest",
            os.path.join(self.experiment.exported_model_dir, "saved_model"),
            do_zip=True,
        )

    def _download_either_original_or_finetuned_artifact(self):
        try:
            checkpoint_file = self.experiment.get_artifact("encoder_weights")
            loaded_checkpoint_path = os.path.join(
                self.experiment.checkpoint_dir, checkpoint_file.filename
            )
            checkpoint_file.download(self.experiment.checkpoint_dir)

        except ResourceNotFoundError:
            loaded_checkpoint_path = None

        try:
            finetuned_checkpoint_file = self.experiment.get_artifact(
                "finetuned_model_weights"
            )
            loaded_finetuned_checkpoint_path = os.path.join(
                self.experiment.checkpoint_dir,
                "finetuned",
                finetuned_checkpoint_file.filename,
            )
            finetuned_checkpoint_file.download(
                os.path.join(self.experiment.checkpoint_dir, "finetuned")
            )
            loaded_checkpoint_path = None

        except ResourceNotFoundError:
            loaded_finetuned_checkpoint_path = None

        return loaded_checkpoint_path, loaded_finetuned_checkpoint_path

    def _get_total_loss(self) -> sm.base.objects.SumOfLosses:
        dice_loss = sm.losses.DiceLoss()
        focal_loss = (
            sm.losses.BinaryFocalLoss()
            if self.n_classes == 1
            else sm.losses.CategoricalFocalLoss()
        )
        total_loss = dice_loss + (1 * focal_loss)
        return total_loss

    def eval(self):
        self.eval_dataset_version = self.experiment.get_dataset(
            name=self.image_dataset_name
        )
        self.label = self.eval_dataset_version.list_labels()[0]
        self.labelmap = {"0": self.label.name}
        self.model.load_weights(self.best_model_path)
        scores = self.model.evaluate(self.eval_dataloader)
        format_and_log_eval_metrics(self.experiment, self.metrics, scores)
        predict_and_log_mask(
            dataset=self.eval_dataset, experiment=self.experiment, model=self.model
        )
        self.setup_files_for_evaluation()
        self.draw_ground_truth_polygons_from_masks()
        self._run_evaluations()
        self.experiment.compute_evaluations_metrics(InferenceType.SEGMENTATION)

    def setup_files_for_evaluation(self):
        for filepath in [self.x_eval_dir, self.y_eval_dir]:
            move_files_for_polygon_creation(
                label_name=self.label.name, input_folder_path=filepath
            )

    def draw_ground_truth_polygons_from_masks(self):
        dataset = self.experiment.get_dataset(self.image_dataset_name)
        dataset.set_type(InferenceType.SEGMENTATION)
        converter = CustomConverter(
            images_dir=self.x_eval_dir,
            masks_dir=self.y_eval_dir,
            labelmap=self.labelmap,
            conversion_tolerance=0.9,
            min_contour_points=20,
        )
        coco_annotations_object = converter.update_coco_annotations()
        coco_annotations_path = os.path.join(
            self.experiment.png_dir, "coco_annotations.json"
        )
        coco_annotations_object.save_coco_annotations_as_json(
            json_path=coco_annotations_path
        )
        dataset.import_annotations_coco_file(
            file_path=coco_annotations_path,
            force_create_label=True,
            fail_on_asset_not_found=False,
        )

    def _run_evaluations(self):
        for i in range(self.eval_dataset.__len__()):
            image_filepath, asset = find_asset_by_dataset_index(
                dataset=self.eval_dataset,
                dataset_version=self.eval_dataset_version,
                i=i,
            )
            if asset is not None:
                image, ground_truth_mask = self.eval_dataset[i]
                predicted_mask = predict_mask_from_image(
                    image=image, model=self.model, asset=asset
                )
                polygons = self._convert_mask_to_polygons(predicted_mask)
                formatted_polygons = format_polygons(polygons=polygons)

                to_send = [
                    (formatted_polygon, self.label, 0.0)
                    for formatted_polygon in formatted_polygons
                ]
                self.experiment.add_evaluation(asset=asset, polygons=to_send)
                logging.info(
                    f"Asset: {get_filename_from_fullpath(image_filepath)} evaluated."
                )

    def _convert_mask_to_polygons(self, mask: np.ndarray) -> list[np.ndarray]:
        polygons = []
        mask_formatted = mask.squeeze()
        contours = find_contours(mask_formatted)
        for contour in contours:
            approximated_contour = approximate_polygon(
                coords=contour, tolerance=self.conversion_tolerance
            )
            if len(approximated_contour) > self.min_contour_points:
                shifted_contour = shift_x_and_y_coordinates(approximated_contour)
                polygons.append(shifted_contour)
        return polygons

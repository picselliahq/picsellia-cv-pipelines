from picsellia import Client
from uuid import UUID
from picsellia.exceptions import (
    ResourceNotFoundError,
    InsufficientResourcesError,
    PicselliaError,
)
from picsellia.types.enums import InferenceType
from picsellia.sdk.model import ModelVersion
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from typing import List, Tuple
import tqdm
import zipfile
import os
from .formatters import TensorflowFormatter
from PIL import Image
import numpy as np
import requests
import logging
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


class Evaluator:
    """ """

    def __init__(
        self,
        client: Client,
        dataset_version_id: UUID,
        model_version_id: UUID,
        experiment_id: UUID,
        parameters: dict = dict(),
    ) -> None:
        self.client = client
        self.experiment = self.client.get_experiment_by_id(experiment_id)
        self.dataset_object: DatasetVersion = self.client.get_dataset_version_by_id(
            dataset_version_id
        )
        self.model_object: ModelVersion = self.client.get_model_version_by_id(
            model_version_id
        )
        self.parameters = parameters

    # Coherence Checks

    def _type_coherence_check(self):
        assert self.dataset_object.type == self.model_object.type, PicselliaError(
            f"Can't run pre-annotation job on a {self.dataset_object.type} with {self.model_object.type} model."
        )

    def _labels_coherence_check(self) -> bool:
        """
        Assert that at least one label from the model labelmap is contained in the dataset version.
        """
        self.model_labels_name = self._get_model_labels_name()
        self.dataset_labels_name = [
            label.name for label in self.dataset_object.list_labels()
        ]

        intersecting_labels = set(self.model_labels_name).intersection(
            self.dataset_labels_name
        )
        logging.info(
            f"Pre-annotation Job will only run on classes: {list(intersecting_labels)}"
        )
        return len(intersecting_labels) > 0

    def _dataset_inclusion_check(
        self,
    ) -> None:
        """
        Check if the selected dataset is included into the given experiment,

        If the dataset isn't in the experiment, we'll add it under the name "eval".
        """

        attached_datasets = self.experiment.list_attached_dataset_versions()
        inclusion = False
        for dataset_version in attached_datasets:
            if dataset_version.id == self.dataset_object.id:
                inclusion = True

        if not inclusion:
            self.experiment.attach_dataset(
                name="eval", dataset_version=self.dataset_object
            )
            logging.info(
                f"{self.dataset_object.name}/{self.dataset_object.version} attached to the experiment."
            )
        return

    # Sanity check

    def _check_model_file_sanity(
        self,
    ) -> None:
        try:
            self.model_object.get_file("model-latest")
        except ResourceNotFoundError:
            raise ResourceNotFoundError(
                "Can't run a pre-annotation job with this model, expected a 'model-latest' file"
            )

    def _check_model_type_sanity(
        self,
    ) -> None:
        if self.model_object.type == InferenceType.NOT_CONFIGURED:
            raise PicselliaError(
                f"Can't run pre-annotation job, {self.model_object.name} type not configured."
            )

    def model_sanity_check(
        self,
    ) -> None:
        self._check_model_file_sanity()
        logging.info(f"Model {self.model_object.name} is sane.")

    # Utilities

    def _is_labelmap_starting_at_zero(
        self,
    ) -> bool:
        return "0" in self.model_infos["labels"].keys()

    def _set_dataset_version_type(
        self,
    ) -> None:
        self.dataset_object.set_type(self.model_object.type)
        logging.info(
            f"Setting dataset {self.dataset_object.name}/{self.dataset_object.version} to type {self.model_object.type}"
        )

    def _get_model_labels_name(
        self,
    ) -> List[str]:
        self.model_infos = self.model_object.sync()
        if "labels" not in self.model_infos.keys():
            raise InsufficientResourcesError(
                f"Can't find labelmap for model {self.model_object.name}"
            )
        if not isinstance(self.model_infos["labels"], dict):
            raise InsufficientResourcesError(
                f"Invalid LabelMap type, expected 'dict', got {type(self.model_infos['labels'])}"
            )
        model_labels = list(self.model_infos["labels"].values())
        return model_labels

    def _create_labels(
        self,
    ) -> None:
        if not hasattr(self, "model_labels_name"):
            self.model_labels_name = self._get_model_labels_name()
        for label in tqdm.tqdm(self.model_labels_name):
            self.dataset_object.create_label(name=label)
        self.dataset_labels_name = [
            label.name for label in self.dataset_object.list_labels()
        ]
        logging.info(f"Labels :{self.dataset_labels_name} created.")

    def _download_model_weights(
        self,
    ):
        model_weights = self.model_object.get_file("model-latest")
        model_weights.download()
        weights_zip_path = model_weights.filename
        with zipfile.ZipFile(weights_zip_path, "r") as zip_ref:
            zip_ref.extractall("saved_model")
        cwd = os.getcwd()
        self.model_weights_path = os.path.join(cwd, "saved_model")
        logging.info(
            f"{self.model_object.name}/{self.model_object.version} weights downloaded."
        )

    def _load_tensorflow_saved_model(
        self,
    ):
        try:
            self.model = tf.saved_model.load(self.model_weights_path)
            logging.info("Model loaded in memory.")
            try:
                self.model = self.model.signatures[
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                ]
                self.input_width, self.input_height = (
                    self.model.inputs[0].shape[1],
                    self.model.inputs[0].shape[2],
                )
                self.ouput_names = list(self.model.structured_outputs.keys())
            except Exception:
                self.input_width, self.input_height = None, None
                self.ouput_names = None
        except Exception:
            raise PicselliaError(
                f"Impossible to load saved model located at: {self.model_weights_path}"
            )

    def setup_preannotation_job(
        self,
    ):
        logging.info(
            f"Setting up the Pre-annotation Job for dataset {self.dataset_object.name}/{self.dataset_object.version} with model {self.model_object.name}/{self.model_object.version}"
        )
        self.model_sanity_check()
        self._dataset_inclusion_check()
        if self.dataset_object.type == InferenceType.NOT_CONFIGURED:
            self._set_dataset_version_type()
            self._create_labels()
        else:
            self._type_coherence_check()
            self._labels_coherence_check()
        self.labels_to_detect = list(
            set(self.model_labels_name).intersection(self.dataset_labels_name)
        )
        self._download_model_weights()
        self._load_tensorflow_saved_model()

    def _preprocess_image(self, asset: Asset) -> np.array:
        image = Image.open(
            requests.get(asset.sync()["data"]["presigned_url"], stream=True).raw
        )
        if self.input_width and self.input_height:
            image = image.resize((self.input_width, self.input_height))
            if image.mode != "RGB":
                image = image.convert("RGB")
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        if self.input_width and self.input_height:
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        return image

    def _format_picsellia_rectangles(
        self, width: int, height: int, predictions: np.array
    ) -> Tuple[List, List, List]:
        formatter = TensorflowFormatter(width, height, self.ouput_names)
        formated_output = formatter.format_object_detection(predictions)
        scores = formated_output["detection_scores"]
        boxes = formated_output["detection_boxes"]
        classes = formated_output["detection_classes"]
        return (scores, boxes, classes)

    def _format_picsellia_polygons(
        self, width: int, height: int, predictions: np.array
    ) -> Tuple[List, List, List, List]:
        formatter = TensorflowFormatter(width, height, self.ouput_names)
        formated_output = formatter.format_segmentation(predictions)
        scores = formated_output["detection_scores"]
        boxes = formated_output["detection_boxes"]
        classes = formated_output["detection_classes"]
        masks = formated_output["detection_masks"]
        return (scores, masks, boxes, classes)

    def _format_and_add_rectangles_evaluation(
        self, asset: Asset, predictions: dict, confidence_treshold: float = 0.5
    ) -> None:
        scores, boxes, classes = self._format_picsellia_rectangles(
            width=asset.width, height=asset.height, predictions=predictions
        )
        #  Convert predictions to Picsellia format
        rectangle_list = []
        nb_box_limit = 100
        if len(boxes) < nb_box_limit:
            nb_box_limit = len(boxes)
        if len(boxes) == 0:
            return
        for i in range(nb_box_limit):
            if scores[i] >= self.parameters.get("confidence_threshold", 0.4):
                try:
                    if self._is_labelmap_starting_at_zero():
                        label = self.dataset_object.get_label(
                            name=self.model_infos["labels"][str(int(classes[i]) - 1)]
                        )
                    else:
                        label = self.dataset_object.get_label(
                            name=self.model_infos["labels"][str(int(classes[i]))]
                        )
                    box = boxes[i]
                    box.append(label)
                    box.append(scores[i])
                    rectangle_list.append(tuple(box))
                except ResourceNotFoundError as e:
                    print(e)
                    continue
        if len(rectangle_list) > 0:
            self.experiment.add_evaluation(asset=asset, rectangles=rectangle_list)
            logging.info(f"Asset: {asset.filename} evaluated.")

    def _format_and_add_polygons_evaluation(
        self, asset: Asset, predictions: dict, confidence_treshold: float
    ) -> None:
        scores, masks, _, classes = self._format_picsellia_polygons(
            width=asset.width, height=asset.height, predictions=predictions
        )
        #  Convert predictions to Picsellia format
        polygons_list = []
        nb_polygons_limit = 100
        if len(masks) < nb_polygons_limit:
            nb_box_limit = len(masks)
        if len(masks) > 0:
            asset.create_annotation(duration=0.0)
        else:
            return
        for i in range(nb_box_limit):
            if scores[i] >= self.parameters.get("confidence_threshold", 0.4):
                try:
                    if self._is_labelmap_starting_at_zero():
                        label = self.dataset_object.get_label(
                            name=self.model_infos["labels"][str(int(classes[i]) - 1)]
                        )
                    else:
                        label = self.dataset_object.get_label(
                            name=self.model_infos["labels"][str(int(classes[i]))]
                        )
                    polygons_list.append((masks[i], label, scores[i]))
                except ResourceNotFoundError as e:
                    print(e)
                    continue
        if len(polygons_list) > 0:
            self.experiment.add_evaluation(asset=asset, polygons=polygons_list)
            logging.info(f"Asset: {asset.filename} evaluated.")

    def preannotate(self, confidence_treshold: float = 0.5):
        dataset_size = self.dataset_object.sync()["size"]
        if "batch_size" not in self.parameters:
            batch_size = 8
        else:
            batch_size = self.parameters["batch_size"]
        batch_size = batch_size if dataset_size > batch_size else dataset_size
        total_batch_number = self.dataset_object.sync()["size"] // batch_size
        for batch_number in tqdm.tqdm(range(total_batch_number)):
            for asset in self.dataset_object.list_assets(
                limit=batch_size, offset=batch_number * batch_size
            ):
                self.experiment.add_evaluations()
                image = self._preprocess_image(asset)
                try:
                    predictions = self.model(image)  # Predict
                except Exception as e:
                    print(e)
                    self.model = tf.saved_model.load(self.model_weights_path)
                    predictions = self.model(image)
                if len(predictions) > 0:
                    #  Format the raw output
                    if self.dataset_object.type == InferenceType.OBJECT_DETECTION:
                        self._format_and_add_rectangles_evaluation(asset, predictions)
                    elif self.dataset_object.type == InferenceType.SEGMENTATION:
                        self._format_and_add_polygons_evaluation(
                            asset, predictions, confidence_treshold
                        )
        self.experiment.run_evaluations(inference_type=self.dataset_object.type)

        #  Fetch original annotation and shapes to overlay over predictions

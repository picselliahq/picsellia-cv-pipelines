import logging
import math
import os
from abc import ABC, abstractmethod
from typing import List, Type, Union, Optional

import numpy as np
import tqdm
from picsellia import Job
from picsellia.exceptions import InsufficientResourcesError, ResourceNotFoundError
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.types.enums import InferenceType

from evaluator.framework_formatter import FrameworkFormatter
from evaluator.type_formatter import TypeFormatter


def _labels_coherence_check(experiment_labelmap, dataset_labels) -> bool:
    """
    Assert that at least one label from the model labelmap is contained in the dataset version.
    """
    model_labels_name = list(experiment_labelmap.values())
    dataset_labels_name = list(dataset_labels.keys())
    intersecting_labels = set(model_labels_name).intersection(dataset_labels_name)
    logging.info(
        f"Pre-annotation Job will only run on classes: {list(intersecting_labels)}"
    )
    return len(intersecting_labels) > 0


class AbstractEvaluator(ABC):
    type_formatter: Type[TypeFormatter]
    framework_formatter: Type[FrameworkFormatter]

    def __init__(
        self,
        experiment: Experiment,
        dataset: DatasetVersion,
        asset_list: Optional[List[Asset]] = None,
        confidence_threshold: float = 0.1,
        weights_path: Optional[str] = None,
    ) -> None:
        self._experiment = experiment
        self._dataset = dataset
        self._parameters = experiment.get_log("parameters").data

        self._asset_list = asset_list if asset_list else self._dataset.list_assets()
        self._batch_size = (
            self._parameters.get("evaluation_batch_size", 8)
            if asset_list
            and len(asset_list) > self._parameters.get("evaluation_batch_size", 8)
            else len(asset_list)
            if asset_list
            else 0
        )

        self._confidence_threshold = confidence_threshold
        self.do_download = True
        if not weights_path:
            self._model_weights = self._experiment.get_artifact(
                self._get_model_artifact_filename()
            )
            self._model_weights_path = os.path.join(
                os.path.join(os.getcwd(), "saved_model"), self._model_weights.filename
            )
        else:
            self.do_download = False
            self._model_weights_path = weights_path
        self._loaded_model = None
        self._nb_object_limit = 100

        self.labelmap = self._setup_label_map()
        self._framework_formatter = self.framework_formatter(labelmap=self.labelmap)
        self._type_formatter = self.type_formatter(
            framework_formatter=self._framework_formatter
        )
        self._setup_evaluation_job()

    def _setup_label_map(self):
        experiment_labelmap = self._get_experiment_labelmap()
        dataset_labels = {label.name: label for label in self._dataset.list_labels()}
        _labels_coherence_check(experiment_labelmap, dataset_labels)

        return {
            int(category_id): dataset_labels[label_name]
            for category_id, label_name in experiment_labelmap.items()
        }

    @abstractmethod
    def _get_model_weights_path(self):
        pass

    def _get_experiment_labelmap(self) -> dict:
        try:
            return self._experiment.get_log("labelmap").data
        except Exception:
            raise InsufficientResourcesError("Can't find labelmap for this experiment")

    def _setup_evaluation_job(self):
        logging.info("Setting up the evaluation for this experiment")
        self._model_sanity_check()
        self._dataset_inclusion_check()
        if self.do_download:
            self._download_model_weights()
            self._get_model_weights_path()
        self._load_saved_model()

    def _model_sanity_check(self) -> None:
        try:
            self._experiment.get_artifact(self._get_model_artifact_filename())
            logging.info(f"Experiment {self._experiment.name} is sane.")
        except ResourceNotFoundError as e:
            raise ResourceNotFoundError(
                "Can't run a pre-annotation job with this model, expected a 'checkpoint-index-latest' file"
            ) from e

    def _dataset_inclusion_check(self) -> None:
        """
        Check if the selected dataset is included into the given experiment,

        If the dataset isn't in the experiment, we'll add it under the name "eval".
        """

        attached_datasets = self._experiment.list_attached_dataset_versions()
        for dataset_version in attached_datasets:
            if dataset_version.id == self._dataset.id:
                return

        self._experiment.attach_dataset(name="eval", dataset_version=self._dataset)
        logging.info(
            f"{self._dataset.name}/{self._dataset.version} attached to the experiment."
        )

    def _download_model_weights(self) -> None:
        self._model_weights.download(
            target_path=os.path.split(self._model_weights_path)[0]
        )
        logging.info("experiment weights downloaded.")

    @abstractmethod
    def _load_saved_model(self):
        pass

    def evaluate(self) -> Job:
        total_batch_number = math.ceil(len(self._asset_list) / self._batch_size)

        for i in tqdm.tqdm(range(total_batch_number)):
            asset_list = self._asset_list[
                i * self._batch_size : (i + 1) * self._batch_size
            ]
            self._evaluate_asset_list(asset_list)
        if self._dataset.type in [
            InferenceType.OBJECT_DETECTION,
            InferenceType.SEGMENTATION,
            InferenceType.CLASSIFICATION,
        ]:
            return self._experiment.compute_evaluations_metrics(
                inference_type=self._dataset.type
            )

    def _evaluate_asset_list(self, asset_list: List[Asset]) -> None:
        if not self._loaded_model:
            raise ValueError("Model not loaded.")
        inputs = self._preprocess_images(asset_list)
        predictions = self._loaded_model(inputs)  # Predict
        for i, asset in enumerate(asset_list):
            evaluations = self._format_prediction_to_evaluations(
                asset=asset, prediction=predictions[i]
            )
            self._send_evaluations_to_platform(asset=asset, evaluations=evaluations)

    @abstractmethod
    def _preprocess_images(self, assets: List[Asset]) -> List[np.ndarray]:
        pass

    #
    # @abstractmethod
    # def _preprocess_image(self, asset: Asset) -> np.ndarray:
    #     pass

    def _format_prediction_to_evaluations(
        self, asset: Asset, prediction: Union[List, dict]
    ) -> List:
        picsellia_predictions = self._type_formatter.format_prediction(
            asset=asset, prediction=prediction
        )

        evaluations = []
        for i in range(
            min(self._nb_object_limit, len(picsellia_predictions["confidences"]))
        ):
            if picsellia_predictions["confidences"][i] >= self._confidence_threshold:
                picsellia_prediction = {
                    prediction_key: prediction[i]
                    for prediction_key, prediction in picsellia_predictions.items()
                }
                evaluation = self._type_formatter.format_evaluation(
                    picsellia_prediction=picsellia_prediction
                )
                evaluations.append(evaluation)
        return evaluations

    def _send_evaluations_to_platform(self, asset: Asset, evaluations: List) -> None:
        shapes = {self._type_formatter.get_shape_type(): evaluations}
        try:
            self._experiment.add_evaluation(asset=asset, **shapes)
            print(f"Asset: {asset.filename} evaluated.")
            logging.info(f"Asset: {asset.filename} evaluated.")
        except Exception:
            logging.info(
                f"Something went wrong with evaluating {asset.filename}. Skipping.."
            )

    def _get_model_artifact_filename(self):
        pass

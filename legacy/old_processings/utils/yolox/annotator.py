import math
from collections import defaultdict
from typing import Dict

import requests
from PIL import Image
from picsellia import Asset
from picsellia.exceptions import (
    ResourceNotFoundError,
    InsufficientResourcesError,
    PicselliaError,
)
from picsellia.types.enums import InferenceType

import tqdm
import os
import cv2
import numpy as np

import logging
import onnxruntime as ort


class YoloxInferenceResult:
    def __init__(
        self, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray
    ) -> None:
        self.boxes = boxes
        self.scores = scores
        self.labels = labels


class PreAnnotator:
    def __init__(self, client, dataset_version_id, model_version_id, parameters):
        self.client = client
        self.dataset_version = client.get_dataset_version_by_id(dataset_version_id)
        self.model_version = client.get_model_version_by_id(model_version_id)
        self.parameters = parameters

        self.model_labels = []
        self.dataset_labels = []
        self.model_info = {}

    def setup_pre_annotation_job(self):
        """
        Set up the pre-annotation job by performing various checks and preparing the model.
        """
        logging.info("Setting up pre-annotation job...")
        self._model_sanity_check()

        if self.dataset_version.type == InferenceType.NOT_CONFIGURED:
            self._set_dataset_type_to_model_type()
            self._create_labels_in_dataset()

        self._validate_dataset_and_model_type()
        self._validate_label_overlap()
        self._download_model_weights()
        self._load_model()

    def pre_annotate(self, confidence_threshold: float = 0.5):
        """
        Processes and annotates assets in the dataset using the YOLOX model.

        Args:
            confidence_threshold (float, optional): A threshold value used to filter
                                                    the bounding boxes based on their
                                                    confidence scores. Only boxes with
                                                    confidence scores above this threshold
                                                    are annotated. Defaults to 0.5.
        """
        dataset_size = self.dataset_version.sync()["size"]
        batch_size = self.parameters.get("batch_size", 8)
        batch_size = min(batch_size, dataset_size)

        total_batch_number = math.ceil(dataset_size / batch_size)

        logging.info(
            f"\n-- Starting processing {total_batch_number} batch(es) of {batch_size} image(s) | "
            f"Total images: {dataset_size} --"
        )

        for batch_number in tqdm.tqdm(
            range(total_batch_number),
            desc="Processing batches",
            unit="batch",
        ):
            offset = batch_number * batch_size
            assets = self.dataset_version.list_assets(limit=batch_size, offset=offset)

            for asset in assets:
                image_data = self._convert_picsellia_asset_to_numpy(asset=asset)
                yolox_inference_result = self._make_model_inference(
                    image_data=image_data
                )

                if yolox_inference_result and len(yolox_inference_result.boxes) > 0:
                    if self.dataset_version.type == InferenceType.OBJECT_DETECTION:
                        self._format_and_save_rectangles(
                            asset, yolox_inference_result, confidence_threshold
                        )
                    else:
                        logging.error(
                            f"The dataset version's type {self.dataset_version.type} is not handled yet "
                            f"by this processing."
                        )

            logging.info("")

    def _convert_picsellia_asset_to_numpy(self, asset: Asset) -> np.ndarray:
        """
        Preprocess a Picsellia asset into a numpy array.

        Args:
            asset (Asset): The Picsellia Asset object containing information and metadata of the asset to be processed.

        Returns:
            np.array: A numpy array representing the downloaded image.
        """
        image = Image.open(
            requests.get(asset.sync()["data"]["presigned_url"], stream=True).raw
        )
        return np.array(image)

    def _format_and_save_rectangles(
        self,
        asset: Asset,
        yolox_inference_result: YoloxInferenceResult,
        confidence_threshold: float = 0.1,
    ) -> None:
        """
        Formats the results of YOLOX inference and saves them as rectangular annotations in the Picsellia platform.

        Args:
            asset (Asset): The Picsellia Asset object to which the annotations will be saved.
            yolox_inference_result (YoloxInferenceResult): The result from YOLOX model inference, containing boxes, scores, and labels.
            confidence_threshold (float, optional): The threshold for filtering predictions based on their confidence scores.
                                                     Defaults to 0.1.
        """
        boxes = yolox_inference_result.boxes
        scores = yolox_inference_result.scores
        labels = yolox_inference_result.labels

        annotation_indexes: Dict[str, int] = defaultdict(int)

        rectangles_to_save = []
        num_boxes = min(len(boxes), 100)

        logging_lines = [f"\nProcessing asset '{asset.filename}':"]

        if len(asset.list_annotations()) > 0:
            logging_lines.append("Asset is already annotated. Skipping.")
            logging.info("\n".join(logging_lines))
            return

        annotation = asset.create_annotation(duration=0.0)

        for i in range(num_boxes):
            if scores[i] < confidence_threshold:
                continue
            try:
                label_name = self.model_labels[int(labels[i])]
                label = self.dataset_version.get_label(name=label_name)
                x_min, y_min, x_max, y_max = boxes[i]
                rectangle = [
                    max(0, int(x_min)),
                    max(0, int(y_min)),
                    int(x_max - x_min),
                    int(y_max - y_min),
                    label,
                ]
                rectangles_to_save.append(tuple(rectangle))
                annotation_indexes[label_name] += 1

                logging_lines.append(
                    f"Found {label.name} #{annotation_indexes[label_name]}"
                    f" | score: {scores[i]}"
                    f" | coordinates: {rectangle[0:4]}"
                )

            except ResourceNotFoundError:
                continue

        if annotation and rectangles_to_save:
            annotation.create_multiple_rectangles(rectangles_to_save)
            logging_lines.append(
                f"Asset '{asset.filename}' pre-annotated with {len(rectangles_to_save)} rectangles."
            )
        else:
            logging_lines.append(
                f"No object has been found for asset '{asset.filename}'."
            )

        logging.info("\n".join(logging_lines))

    def _model_sanity_check(self) -> None:
        """
        Perform a sanity check on the model.
        """
        self._check_model_file_integrity()
        self._validate_model_inference_type()
        logging.info(f"Model {self.model_version.name} passed sanity checks.")

    def _check_model_file_integrity(self) -> None:
        """
        Check the integrity of the model file by verifying it exists as "model-latest" and is an ONNX model.

        Raises:
            ResourceNotFoundError: If the model file is not an ONNX file.
        """
        model_file = self.model_version.get_file("model-latest")
        if not model_file.filename.endswith(".onnx"):
            raise ResourceNotFoundError("Model file must be an ONNX file.")

    def _validate_model_inference_type(self) -> None:
        """
        Validate the model's inference type.

        Raises:
            PicselliaError: If the model type is not configured.
        """
        if self.model_version.type == InferenceType.NOT_CONFIGURED:
            raise PicselliaError("Model type is not configured.")

    def _set_dataset_type_to_model_type(self) -> None:
        """
        Set the dataset type to the model type.
        """
        self.dataset_version.set_type(self.model_version.type)
        logging.info(f"Dataset type set to {self.model_version.type}")

    def _create_labels_in_dataset(self) -> None:
        """
        Creates labels in the dataset based on the model's labels. It first retrieves the model's labels,
        then creates corresponding labels in the dataset version if they do not already exist.

        This method updates the 'model_labels' and 'dataset_labels' attributes of the class with the
        labels from the model (if they don't already exist) and the labels currently in the dataset, respectively.
        """
        if not self.model_labels:
            self.model_labels = self._get_model_labels()

        for label in tqdm.tqdm(self.model_labels):
            self.dataset_version.create_label(name=label)

        self.dataset_labels = [
            label.name for label in self.dataset_version.list_labels()
        ]
        logging.info(f"Labels created in dataset: {self.dataset_labels}")

    def _validate_dataset_and_model_type(self) -> None:
        """
        Validate that the dataset type matches the model type.

        Raises:
            PicselliaError: If the dataset type does not match the model type.
        """
        if self.dataset_version.type != self.model_version.type:
            raise PicselliaError(
                f"Dataset type {self.dataset_version.type} does not match model type {self.model_version.type}."
            )

    def _validate_label_overlap(self) -> None:
        """
        Validate that there is an overlap between model labels and dataset labels.

        Raises:
            PicselliaError: If no overlapping labels are found.
        """
        self.model_labels = self._get_model_labels()
        self.dataset_labels = [
            label.name for label in self.dataset_version.list_labels()
        ]

        model_labels_set = set(self.model_labels)
        dataset_labels_set = set(self.dataset_labels)

        overlapping_labels = model_labels_set.intersection(dataset_labels_set)
        non_overlapping_dataset_labels = dataset_labels_set - model_labels_set

        if not overlapping_labels:
            raise PicselliaError(
                "No overlapping labels found between model and dataset. "
                "Please check the labels between your dataset version and your model.\n"
                f"Dataset labels: {self.dataset_labels}\n"
                f"Model labels: {self.model_labels}"
            )

        # Log the overlapping labels
        logging.info(f"Using labels: {list(overlapping_labels)}")

        if non_overlapping_dataset_labels:
            logging.info(
                f"WARNING: Some dataset version's labels are not present in model "
                f"and will be skipped: {list(non_overlapping_dataset_labels)}"
            )

    def _download_model_weights(self) -> None:
        """
        Download the model weights and save it in `self.model_weights_path`.
        """
        model_weights = self.model_version.get_file("model-latest")
        model_weights.download()
        self.model_weights_path = os.path.join(os.getcwd(), model_weights.filename)
        logging.info(f"Model weights {model_weights.filename} downloaded successfully")

    def _load_model(self) -> None:
        """
        Load the model from the downloaded weights.

        Raises:
            PicselliaError: If there is an error in loading the model.
        """
        try:
            import onnx

            self.model = onnx.load(self.model_weights_path)
            onnx.checker.check_model(self.model)

            self.ort_session = ort.InferenceSession(self.model_weights_path)
            logging.info("Model loaded successfully.")
        except Exception as e:
            raise PicselliaError(f"Error loading model: {e}")

    def _get_model_labels(self) -> list[str]:
        """
        Get the labels from the model.

        Returns:
            list[str]: A list of label names from the model.
        Raises:
            InsufficientResourcesError: If no labels are found or if labels are not in dictionary format.
        """
        self.model_info = self.model_version.sync()
        if "labels" not in self.model_info:
            raise InsufficientResourcesError(
                f"No labels found for model {self.model_version.name}."
            )

        if not isinstance(self.model_info["labels"], dict):
            raise InsufficientResourcesError(
                "Model labels must be in dictionary format."
            )

        return list(self.model_info["labels"].values())

    def _is_labelmap_zero_based(self) -> bool:
        """
        Check if the label map starts at zero.

        Returns:
            bool: True if the label map starts at zero, False otherwise.
        """
        return "0" in self.model_info["labels"]

    def _prepare_padded_image(
        self, image_data: np.ndarray, input_size: tuple[int, int]
    ) -> tuple[np.ndarray, float]:
        """
        Prepare a padded image for model inference. The image is resized based on the input size
        and padded with a constant value if necessary to match the input size.

        Args:
            image_data (np.ndarray): The original image.
            input_size (tuple[int, int]): Desired input size for the model.

        Returns:
            tuple[np.ndarray, float]: Padded image and the resize ratio.
        """

        image_data = self.convert_to_rgb(image_data=image_data)

        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        image_ratio = min(
            input_size[0] / image_data.shape[0], input_size[1] / image_data.shape[1]
        )
        resized_image = self._resize_image(image_data, image_ratio)
        padded_img[
            : int(image_data.shape[0] * image_ratio),
            : int(image_data.shape[1] * image_ratio),
        ] = resized_image
        return padded_img, image_ratio

    def convert_to_rgb(self, image_data: np.ndarray) -> np.ndarray:
        """
        Convert an image to RGB format if it's not already in RGB.

        Args:
            image_data (np.ndarray): The original image data. Can be in various formats including RGBA and CMYK.

        Returns:
            np.ndarray: The image data converted to RGB format.
        """
        image = Image.fromarray(image_data)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_data = np.array(image)
        image.close()

        return image_data

    def _resize_image(self, img: np.ndarray, ratio: float) -> np.ndarray:
        """
        Resize the input image 'img' based on the given 'ratio'.

        Args:
            img (np.ndarray): The input image as a NumPy array.
            ratio (float): The ratio by which the image should be resized.

        Returns:
            np.ndarray: The resized image as a NumPy array.
        """
        return cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

    def _make_model_inference(
        self, image_data: np.ndarray
    ) -> YoloxInferenceResult | None:
        """
        Perform inference using the loaded model on the provided 'image_data'.

        Args:
            image_data (np.ndarray): The input image data as a NumPy array.

        Returns:
            YoloxInferenceResult | None: The inference results as a YoloxInferenceResult object,
            or None if no valid results are found.
        """
        input_tensor = self.ort_session.get_inputs()[0]
        input_shape = input_tensor.shape
        expected_input_size = (input_shape[2], input_shape[3])

        img, ratio = self._preprocess_input_for_model(image_data, expected_input_size)
        ort_inputs = {self.ort_session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.ort_session.run(None, ort_inputs)
        predictions = self._postprocess_model_output(output[0], expected_input_size)[0]
        return self._extract_inference_results(predictions, ratio)

    def _preprocess_input_for_model(
        self,
        img: np.ndarray,
        input_size: tuple[int, int],
        swap: tuple[int, int, int] = (2, 0, 1),
    ) -> tuple[np.ndarray, float]:
        """
        Preprocess the input image for the model.

        Args:
            img (np.ndarray): The image to preprocess.
            input_size (tuple[int, int]): The input size for the model.
            swap (tuple[int, int, int]): The order to swap the image channels. Default is (2, 0, 1).
        Returns:
            tuple[np.ndarray, float]: The preprocessed image and the resize ratio.
        """
        padded_image, ratio = self._prepare_padded_image(img, input_size)
        padded_image = padded_image.transpose(swap)
        padded_img = np.ascontiguousarray(padded_image, dtype=np.float32)

        return padded_img, ratio

    def _postprocess_model_output(
        self, outputs: np.ndarray, img_size: tuple[int, int], p6: bool = False
    ) -> np.ndarray:
        """
        Postprocess the model outputs.

        Args:
            outputs (np.ndarray): The raw model outputs as a NumPy array.
            img_size (tuple[int, int]): The size of the input image (height, width).
            p6 (bool, optional): Whether the model uses P6 output. Defaults to False.

        Returns:
            np.ndarray: The post-processed model outputs as a NumPy array.
        """
        outputs = self._apply_strides_to_outputs(outputs, img_size, p6)
        return outputs

    def _apply_strides_to_outputs(
        self, outputs: np.ndarray, img_size: tuple[int, int], p6: bool
    ) -> np.ndarray:
        """
        Apply strides to the model outputs to obtain final bounding box coordinates.

        Args:
            outputs (np.ndarray): The raw model outputs as a NumPy array.
            img_size (tuple[int, int]): The size of the input image (height, width).
            p6 (bool): Whether the model uses P6 output.

        Returns:
            np.ndarray: The postprocessed model outputs as a NumPy array.
        """
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        grids, expanded_strides = self._generate_grids_and_strides(img_size, strides)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def _generate_grids_and_strides(
        self, img_size: tuple[int, int], strides: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate grids and expanded strides based on the input image size and strides.

        Args:
            img_size (tuple[int, int]): The size of the input image (height, width).
            strides (list[int]): List of strides for different scales.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing grids and expanded strides as NumPy arrays.
        """
        grids = []
        expanded_strides = []
        for stride in strides:
            hsize, wsize = img_size[0] // stride, img_size[1] // stride
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full(grid.shape[:2] + (1,), stride))
        return np.concatenate(grids, 1), np.concatenate(expanded_strides, 1)

    def _extract_inference_results(
        self, predictions: np.ndarray, ratio: float
    ) -> YoloxInferenceResult | None:
        """
        Extract inference results from the model predictions.

        Args:
            predictions (np.ndarray): The model predictions as a NumPy array.
            ratio (float): The ratio used during preprocessing.

        Returns:
            YoloxInferenceResult | None: The extracted inference results as a YoloxInferenceResult object,
            or None if no valid results are found.
        """
        boxes, scores = predictions[:, :4], predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = self._convert_to_xyxy_format(boxes) / ratio
        dets = self._multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        if dets is not None:
            final_boxes, final_scores, final_labels = (
                dets[:, :4],
                dets[:, 4],
                dets[:, 5],
            )

            return YoloxInferenceResult(
                boxes=final_boxes, scores=final_scores, labels=final_labels
            )

        return None

    def _convert_to_xyxy_format(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding box coordinates from (x_center, y_center, width, height) to (x1, y1, x2, y2) format.

        Args:
            boxes (np.ndarray): Bounding box coordinates in (x_center, y_center, width, height) format.

        Returns:
            np.ndarray: Bounding box coordinates in (x1, y1, x2, y2) format.
        """
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return boxes_xyxy

    def _multiclass_nms(
        self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float, score_thr: float
    ) -> np.ndarray | None:
        """
        Multiclass NMS implemented in Numpy. Class-aware version.

        Args:
            boxes (np.ndarray): Bounding box coordinates in (x1, y1, x2, y2) format.
            scores (np.ndarray): Confidence scores for each class.
            nms_thr (float): IoU threshold for NMS.
            score_thr (float): Score threshold to consider a detection.

        Returns:
            np.ndarray | None: The filtered and non-maximum suppressed bounding boxes as a NumPy array,
            or None if no valid results are found.
        """
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self._nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None

        return np.concatenate(final_dets, 0)

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, nms_thr: float
    ) -> list[np.ndarray]:
        """
        Single class NMS implemented in Numpy.

        Args:
            boxes (np.ndarray): Bounding box coordinates in (x1, y1, x2, y2) format.
            scores (np.ndarray): Confidence scores for each bounding box.
            nms_thr (float): IoU threshold for NMS.

        Returns:
            list[np.ndarray]: Indices of the kept bounding boxes after NMS.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

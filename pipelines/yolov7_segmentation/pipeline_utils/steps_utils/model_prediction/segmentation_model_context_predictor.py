import os
import shutil
import subprocess

import cv2
from picsellia_cv_engine.models.data.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.model.picsellia_prediction import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)

from pipelines.yolov7_segmentation.pipeline_utils.model.yolov7_model_context import (
    Yolov7ModelContext,
    find_latest_run_dir,
)
from pipelines.yolov7_segmentation.pipeline_utils.parameters.yolov7_hyper_parameters import (
    Yolov7HyperParameters,
)


class Yolov7SegmentationModelContextPredictor:
    def __init__(self, model_context: Yolov7ModelContext):
        """
        Initialize the inference runner with model context and experiment.

        Args:
            model_context (ModelContext): Context of the model including configuration and weights.
        """
        self.model_context = model_context

    def pre_process_dataset_context(
        self, dataset_context: TBaseDatasetContext
    ) -> list[str]:
        """
        Prepares the dataset by extracting and returning a list of image file paths from the dataset context.

        Args:
            dataset_context (TDatasetContext): The context containing the dataset information.

        Returns:
            List[str]: A list of image file paths from the dataset.
        """
        if not dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")
        image_paths = [
            os.path.join(dataset_context.images_dir, image_name)
            for image_name in os.listdir(dataset_context.images_dir)
        ]
        return image_paths

    def run_inference(
        self,
        image_paths: list[str],
        hyperparameters: Yolov7HyperParameters,
    ) -> dict[str, list[str]]:
        if not self.model_context.trained_weights_path or not os.path.exists(
            self.model_context.trained_weights_path
        ):
            raise ValueError("Trained weights path is not set.")

        if not self.model_context.results_dir or not os.path.exists(
            self.model_context.results_dir
        ):
            raise ValueError("Results directory is not set.")

        image_batches = self._prepare_batches(image_paths, hyperparameters.batch_size)

        for batch in image_batches:
            tmp_dir = os.path.abspath("tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            for image_path in batch:
                shutil.copy(image_path, tmp_dir)

            project_dir = os.path.join(self.model_context.results_dir, "inference")
            os.makedirs(project_dir, exist_ok=True)

            detect_file_path = os.path.abspath(
                "src/pipelines/yolov7_segmentation/yolov7/seg/segment/predict.py"
            )

            print(
                f"Running inference with weights: {self.model_context.trained_weights_path}"
            )

            command = [
                "python3.10",
                detect_file_path,
                "--weights",
                self.model_context.trained_weights_path,
                "--source",
                tmp_dir,
                "--img-size",
                str(hyperparameters.image_size),
                "--conf-thres",
                str(hyperparameters.confidence_threshold),
                "--iou-thres",
                str(hyperparameters.iou_threshold),
                "--device",
                str(hyperparameters.device),
                "--save-txt",
                "--save-conf",
                "--project",
                project_dir,
                "--name",
                self.model_context.model_name,
                "--exist-ok",
            ]

            print(f"Running command: {' '.join(command)}")

            process = subprocess.Popen(command, stdout=None, stderr=None, text=True)

            return_code = process.wait()
            if return_code != 0:
                print("Inference failed with errors.")
            else:
                print("Inference completed successfully.")

            shutil.rmtree(tmp_dir)

        latest_run = find_latest_run_dir(
            os.path.join(self.model_context.results_dir, "inference")
        )

        labels_dir = os.path.join(
            self.model_context.results_dir, "inference", latest_run, "labels"
        )
        mask_dir = os.path.join(
            self.model_context.results_dir, "inference", latest_run, "masks"
        )

        labels_paths = [
            os.path.join(labels_dir, label_name)
            for label_name in os.listdir(labels_dir)
        ]
        mask_paths = [
            os.path.join(mask_dir, mask_name) for mask_name in os.listdir(mask_dir)
        ]

        label_path_to_mask_paths: dict[str, list[str]] = {
            label_filepath: [] for label_filepath in labels_paths
        }

        for mask_path in mask_paths:
            label_name = os.path.basename(mask_path).split("_")[0] + ".txt"
            label_filepath = os.path.join(labels_dir, label_name)
            label_path_to_mask_paths[label_filepath].append(mask_path)

        return label_path_to_mask_paths

    def _prepare_batches(
        self, image_paths: list[str], batch_size: int
    ) -> list[list[str]]:
        """
        Divides the list of image paths into smaller batches of a specified size.

        Args:
            image_paths (List[str]): A list of image file paths to be split into batches.
            batch_size (int): The size of each batch.

        Returns:
            List[List[str]]: A list of batches, each containing a list of image file paths.
        """
        return [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

    def post_process(
        self,
        label_path_to_mask_paths: dict[str, list[str]],
        dataset_context: TBaseDatasetContext,
    ) -> list[PicselliaPolygonPrediction]:
        """
        Post-processes the predictions for a segmentation model, mapping polygons and confidence scores
        to Picsellia assets and labels.

        Args:
            label_path_to_mask_paths (Dict[str, List[str]]): Mapping of label files to associated mask paths.
            dataset_context (TDatasetContext): The dataset context for label and asset mapping.

        Returns:
            List[PicselliaPolygonPrediction]: A list of processed polygon predictions for each asset.
        """
        predictions = []

        for label_path, mask_paths in label_path_to_mask_paths.items():
            asset_id = os.path.basename(label_path).split(".")[0]
            asset = dataset_context.dataset_version.list_assets(ids=[asset_id])[0]

            label_info = self._parse_label_file(label_path)

            polygons = []
            labels = []
            confidences = []

            for mask_index, (class_id, _x1, _y1, _x2, _y2, confidence) in enumerate(
                label_info
            ):
                if mask_index >= len(mask_paths):
                    continue

                polygon = self._extract_largest_contour(
                    mask_paths[mask_index], epsilon=5.0
                )

                if polygon:
                    picsellia_polygon = PicselliaPolygon(points=polygon)
                    polygons.append(picsellia_polygon)

                    picsellia_label = self.get_picsellia_label(
                        class_id, dataset_context
                    )
                    picsellia_confidence = self.get_picsellia_confidence(confidence)

                    labels.append(picsellia_label)
                    confidences.append(picsellia_confidence)

            prediction = PicselliaPolygonPrediction(
                asset=asset, polygons=polygons, labels=labels, confidences=confidences
            )
            predictions.append(prediction)

        return predictions

    def _parse_label_file(self, label_path: str) -> list[tuple]:
        """
        Parse the label file to extract class id, coordinates, and confidence.

        Args:
            label_path (str): Path to the label file.

        Returns:
            List[tuple]: List of tuples (class_id, x1, y1, x2, y2, confidence).
        """
        label_info = []
        with open(label_path) as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                confidence = float(parts[5])
                coordinates = list(map(float, parts[1:5]))
                label_info.append((class_id, *coordinates, confidence))
        return label_info

    def _extract_largest_contour(
        self, mask_path: str, epsilon: float = 3.0
    ) -> list[list[int]]:
        """
        Extract the largest contour from a binary mask image and simplify it if possible.

        Args:
            mask_path (str): Path to the binary mask image.
            epsilon (float): Parameter specifying the approximation accuracy for contour simplification.
                            Higher values result in more simplification.

        Returns:
            List[List[int]]: List of [x, y] coordinates representing the largest contour,
                            simplified if it has at least 4 points, otherwise empty list.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            if len(largest_contour) < 4:
                print(
                    "Warning: Largest contour has fewer than 4 points, it will be ignored."
                )
                return []

            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(simplified_contour) >= 4:
                return [
                    [int(point[0][0]), int(point[0][1])] for point in simplified_contour
                ]
            else:
                print(
                    "Warning: Simplified contour has fewer than 4 points, using the original contour."
                )
                return [
                    [int(point[0][0]), int(point[0][1])] for point in largest_contour
                ]
        else:
            return []

    def get_picsellia_label(
        self, class_id: int, dataset_context: TBaseDatasetContext
    ) -> PicselliaLabel:
        """
        Map the class ID to a PicselliaLabel object using the dataset context.

        Args:
            class_id (int): The class ID.
            dataset_context (TDatasetContext): The dataset context with label mapping.

        Returns:
            PicselliaLabel: The corresponding Picsellia label.
        """
        labels_list = list(dataset_context.labelmap.values())
        return PicselliaLabel(labels_list[class_id])

    def get_picsellia_confidence(self, confidence: float) -> PicselliaConfidence:
        """
        Convert the confidence score to a PicselliaConfidence object.

        Args:
            confidence (float): The confidence score.

        Returns:
            PicselliaConfidence: The corresponding Picsellia confidence object.
        """
        return PicselliaConfidence(value=confidence)

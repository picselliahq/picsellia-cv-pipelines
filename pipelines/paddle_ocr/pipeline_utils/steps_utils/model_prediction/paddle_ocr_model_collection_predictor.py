import os

from picsellia_cv_engine.models.data.dataset.base_dataset import (
    TBaseDataset,
)
from picsellia_cv_engine.models.model.picsellia_prediction import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaOCRPrediction,
    PicselliaRectangle,
    PicselliaText,
)
from picsellia_cv_engine.models.steps.model.predictor.model_collection_predictor import (
    ModelCollectionPredictor,
)

from pipelines.paddle_ocr.pipeline_utils.model.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)


class PaddleOCRModelCollectionPredictor(
    ModelCollectionPredictor[PaddleOCRModelCollection]
):
    """
    A predictor class that handles model inference and result post-processing for OCR tasks
    using the PaddleOCR framework.

    This class performs pre-processing of datasets, runs inference on batches of images, and post-processes
    the predictions to generate PicselliaOCRPrediction objects for OCR tasks.
    """

    def __init__(self, model_collection: PaddleOCRModelCollection):
        """
        Initializes the PaddleOCRModelPredictor with the provided model collection.

        Args:
            model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models.
        """
        super().__init__(model_collection)

    def pre_process_dataset(self, dataset: TBaseDataset) -> list[str]:
        """
        Prepares the dataset by extracting and returning a list of image file paths from the dataset.

        Args:
            dataset (TDataset): The context containing the dataset information.

        Returns:
            List[str]: A list of image file paths from the dataset.
        """
        if not dataset.images_dir:
            raise ValueError("No images directory found in the dataset.")
        image_paths = [
            os.path.join(dataset.images_dir, image_name)
            for image_name in os.listdir(dataset.images_dir)
        ]
        return image_paths

    def prepare_batches(
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

    def run_inference_on_batches(self, image_batches: list[list[str]]) -> list[list]:
        """
        Runs model inference on each batch of images and returns the prediction results for all batches.

        Args:
            image_batches (List[List[str]]): A list of batches of image file paths for inference.

        Returns:
            List[List]: A list of prediction results for each batch.
        """
        all_batch_results = []
        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_paths: list[str]) -> list:
        """
        Executes inference on a single batch of images using the loaded PaddleOCR model.

        Args:
            batch_paths (List[str]): A list of image file paths to perform inference on.

        Returns:
            List: The inference results, containing predictions for each image in the batch.
        """
        batch_results = []
        for image_path in batch_paths:
            result = self.model_collection.loaded_model.ocr(image_path)
            batch_results.append(result)
        return batch_results

    def post_process_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[list],
        dataset: TBaseDataset,
    ) -> list[PicselliaOCRPrediction]:
        """
        Post-processes the inference results for each batch and returns a list of OCR predictions.

        Args:
            image_batches (List[List[str]]): A list of batches of image paths.
            batch_results (List[List]): The list of inference results for each batch.
            dataset (TDataset): The context of the dataset used for label mapping.

        Returns:
            List[PicselliaOCRPrediction]: A list of processed OCR predictions for each image.
        """
        all_predictions = []
        for batch_result, batch_paths in zip(
            batch_results, image_batches, strict=False
        ):
            all_predictions.extend(
                self._post_process(
                    batch_paths=batch_paths,
                    batch_prediction=batch_result,
                    dataset=dataset,
                )
            )
        return all_predictions

    def _post_process(
        self,
        batch_paths: list[str],
        batch_prediction: list,
        dataset: TBaseDataset,
    ) -> list[PicselliaOCRPrediction]:
        """
        Post-processes the predictions for a single batch of images, mapping predicted bounding boxes, texts,
        and confidence scores to PicselliaOCRPrediction objects.

        Args:
            batch_paths (List[str]): The list of image paths for the batch.
            batch_prediction (List): The inference results for the batch.
            dataset (TDataset): The dataset used for label mapping.

        Returns:
            List[PicselliaOCRPrediction]: A list of processed predictions, including image paths, predicted texts,
            bounding boxes, and confidence scores.
        """
        processed_predictions = []

        for image_path, prediction in zip(batch_paths, batch_prediction, strict=False):
            boxes, texts, confidences = self.get_annotations_from_result(prediction)
            asset_id = os.path.basename(image_path).split(".")[0]
            asset = dataset.dataset_version.list_assets(ids=[asset_id])[0]
            labels = [
                PicselliaLabel(dataset.dataset_version.get_or_create_label("text"))
                for _ in texts
                for _ in texts
            ]

            processed_prediction = PicselliaOCRPrediction(
                asset=asset,
                boxes=boxes,
                labels=labels,
                texts=texts,
                confidences=confidences,
            )
            processed_predictions.append(processed_prediction)

        return processed_predictions

    def get_annotations_from_result(
        self, result
    ) -> tuple[
        list[PicselliaRectangle], list[PicselliaText], list[PicselliaConfidence]
    ]:
        """
        Extracts boxes, texts, and confidence scores from a PaddleOCR result.

        Args:
            result: The result from a PaddleOCR prediction.

        Returns:
            Tuple containing lists of PicselliaRectangle, PicselliaText, and PicselliaConfidence.
        """
        result = result[0]
        if not result:
            return [], [], []
        boxes = [self.get_picsellia_rectangle(line[0]) for line in result]
        texts = [self.get_picsellia_text(line[1][0]) for line in result]
        confidences = [self.get_picsellia_confidence(line[1][1]) for line in result]
        return boxes, texts, confidences

    def get_picsellia_rectangle(self, points: list[list[int]]) -> PicselliaRectangle:
        """
        Converts a list of points into a PicselliaRectangle.

        Args:
            points (List[List[int]]): List of points defining the rectangle.

        Returns:
            PicselliaRectangle: The bounding box.
        """
        x = min(point[0] for point in points)
        y = min(point[1] for point in points)
        w = max(point[0] for point in points) - x
        h = max(point[1] for point in points) - y
        return PicselliaRectangle(x, y, w, h)

    def get_picsellia_text(self, text: str) -> PicselliaText:
        """
        Converts text into a PicselliaText object.

        Args:
            text (str): The recognized text.

        Returns:
            PicselliaText: The PicselliaText object.
        """
        return PicselliaText(text)

    def get_picsellia_confidence(self, confidence: float) -> PicselliaConfidence:
        """
        Converts a confidence score into a PicselliaConfidence object.

        Args:
            confidence (float): The confidence score.

        Returns:
            PicselliaConfidence: The PicselliaConfidence object.
        """
        return PicselliaConfidence(confidence)

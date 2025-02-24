import os

from picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.model.picsellia_prediction import (
    PicselliaClassificationPrediction,
)
from picsellia_cv_engine.models.steps.model_prediction.model_context_predictor import (
    ModelContextPredictor,
)
from ultralytics.engine.results import Results

from pipelines.yolov8_classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)


class UltralyticsClassificationModelContextPredictor(
    ModelContextPredictor[UltralyticsModelContext]
):
    """
    A predictor class that handles model inference and result post-processing for classification tasks
    using the Ultralytics framework.

    This class performs pre-processing of datasets, runs inference on batches of images, and post-processes
    the predictions to generate PicselliaClassificationPrediction objects for classification tasks.
    """

    def __init__(self, model_context: UltralyticsModelContext):
        """
        Initializes the UltralyticsClassificationModelContextPredictor with a provided model context.

        Args:
            model_context (ModelContext): The context containing the loaded model and its configurations.
        """
        super().__init__(model_context)

    def pre_process_dataset_context(
        self, dataset_context: TBaseDatasetContext
    ) -> list[str]:
        """
        Prepares the dataset by extracting and returning a list of image file paths from the dataset context.

        Args:
            dataset_context (TBaseDatasetContext): The context containing the dataset information.

        Returns:
            List[str]: A list of image file paths from the dataset.
        """
        if not dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")
        image_paths = []
        for category_name in os.listdir(dataset_context.images_dir):
            category_dir = os.path.join(dataset_context.images_dir, category_name)
            image_paths.extend(
                [
                    os.path.join(category_dir, image_name)
                    for image_name in os.listdir(category_dir)
                ]
            )
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

    def run_inference_on_batches(self, image_batches: list[list[str]]) -> list[Results]:
        """
        Runs model inference on each batch of images and returns the prediction results for all batches.

        Args:
            image_batches (List[List[str]]): A list of batches of image file paths for inference.

        Returns:
            List[Results]: A list of prediction results for each batch.
        """
        all_batch_results = []

        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_paths: list[str]) -> Results:
        """
        Executes inference on a single batch of images using the loaded model.

        Args:
            batch_paths (List[str]): A list of image file paths to perform inference on.

        Returns:
            Results: The inference results, containing predictions for each image in the batch.
        """
        return self.model_context.loaded_model(batch_paths)

    def post_process_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[Results],
        dataset_context: TBaseDatasetContext,
    ) -> list[PicselliaClassificationPrediction]:
        """
        Post-processes the inference results for each batch and returns a list of classification predictions.

        Args:
            image_batches (List[List[str]]): A list of batches of image paths.
            batch_results (List[Results]): The list of inference results for each batch.
            dataset_context (TBaseDatasetContext): The context of the dataset used for label mapping.

        Returns:
            List[PicselliaClassificationPrediction]: A list of processed classification predictions for each image.
        """
        all_predictions = []

        for batch_result, batch_paths in zip(
            batch_results, image_batches, strict=False
        ):
            all_predictions.extend(
                self._post_process(
                    image_paths=batch_paths,
                    batch_prediction=batch_result,
                    dataset_context=dataset_context,
                )
            )
        return all_predictions

    def _post_process(
        self,
        image_paths: list[str],
        batch_prediction: Results,
        dataset_context: TBaseDatasetContext,
    ) -> list[PicselliaClassificationPrediction]:
        """
        Post-processes the predictions for a single batch of images, mapping predicted classes and confidence scores
        to Picsellia labels.

        Args:
            image_paths (List[str]): The list of image paths for the batch.
            batch_prediction (Results): The inference results for the batch.
            dataset_context (TBaseDatasetContext): The dataset context used for label mapping.

        Returns:
            List[PicselliaClassificationPrediction]: A list of processed predictions, including image paths,
            predicted classes, and confidence scores.
        """
        processed_predictions = []

        for image_path, prediction in zip(image_paths, batch_prediction, strict=False):
            asset_id = os.path.basename(image_path).split(".")[0]
            asset = dataset_context.dataset_version.list_assets(ids=[asset_id])[0]
            predicted_label = self.get_picsellia_label(
                prediction.names[int(prediction.probs.top1)], dataset_context
            )
            prediction_confidence = self.get_picsellia_confidence(
                float(prediction.probs.top1conf.cpu().numpy())
            )
            processed_prediction = PicselliaClassificationPrediction(
                asset=asset,
                label=predicted_label,
                confidence=prediction_confidence,
            )
            processed_predictions.append(processed_prediction)

        return processed_predictions

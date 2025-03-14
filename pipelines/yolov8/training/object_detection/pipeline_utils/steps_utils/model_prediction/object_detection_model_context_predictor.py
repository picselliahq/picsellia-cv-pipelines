import os

from picsellia import Asset
from picsellia_cv_engine.models.data.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.model.picsellia_prediction import (
    PicselliaConfidence,
    PicselliaRectangle,
    PicselliaRectanglePrediction,
)
from picsellia_cv_engine.models.steps.model.predictor.model_context_predictor import (
    ModelContextPredictor,
)
from ultralytics.engine.results import Results

from pipelines.yolov8.training.classification.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)


class UltralyticsDetectionModelContextPredictor(
    ModelContextPredictor[UltralyticsModelContext]
):
    """
    A predictor class that handles model inference and result post-processing for object detection tasks
    using the Ultralytics framework.
    """

    def __init__(self, model_context: UltralyticsModelContext):
        super().__init__(model_context)

    def pre_process_dataset_context(
        self, dataset_context: TBaseDatasetContext
    ) -> list[str]:
        if not dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")

        return [
            os.path.join(dataset_context.images_dir, image_name)
            for image_name in os.listdir(dataset_context.images_dir)
        ]

    def prepare_batches(
        self, image_paths: list[str], batch_size: int
    ) -> list[list[str]]:
        return [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

    def run_inference_on_batches(self, image_batches: list[list[str]]) -> list[Results]:
        all_batch_results = []

        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_paths: list[str]) -> Results:
        return self.model_context.loaded_model(batch_paths)

    def post_process_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[Results],
        dataset_context: TBaseDatasetContext,
    ) -> list[PicselliaRectanglePrediction]:
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
    ) -> list[PicselliaRectanglePrediction]:
        processed_predictions = []

        for image_path, prediction in zip(image_paths, batch_prediction, strict=False):
            asset_id = os.path.basename(image_path).split(".")[0]
            asset = dataset_context.dataset_version.list_assets(ids=[asset_id])[0]

            boxes, labels, confidences = self.format_predictions(
                asset=asset, prediction=prediction, dataset_context=dataset_context
            )

            processed_prediction = PicselliaRectanglePrediction(
                asset=asset,
                boxes=boxes,
                labels=labels,
                confidences=confidences,
            )
            processed_predictions.append(processed_prediction)

        return processed_predictions

    def format_predictions(
        self, asset: Asset, prediction: Results, dataset_context: TBaseDatasetContext
    ):
        if not prediction.boxes:
            return [], [], []

        # Extract normalized boxes and rescale them
        normalized_boxes = prediction.boxes.xyxyn.cpu().numpy()
        boxes_list = [
            self.rescale_normalized_box(box, asset.width, asset.height)
            for box in normalized_boxes
        ]
        casted_boxes = [self.cast_type_list_to_int(box) for box in boxes_list]

        # Convert to Picsellia types
        picsellia_boxes = [PicselliaRectangle(*box) for box in casted_boxes]
        picsellia_labels = [
            self.get_picsellia_label(
                prediction.names[int(cls.cpu().numpy())], dataset_context
            )
            for cls in prediction.boxes.cls
        ]
        picsellia_confidences = [
            PicselliaConfidence(float(conf.cpu().numpy()))
            for conf in prediction.boxes.conf
        ]

        return picsellia_boxes, picsellia_labels, picsellia_confidences

    @staticmethod
    def rescale_normalized_box(box, width, height):
        """
        Rescale a normalized bounding box (values between 0 and 1) to image dimensions.
        """
        x_min, y_min, x_max, y_max = box
        return [
            int(x_min * width),
            int(y_min * height),
            int((x_max - x_min) * width),
            int((y_max - y_min) * height),
        ]

    @staticmethod
    def cast_type_list_to_int(box):
        """
        Casts a list of float values to integers.
        """
        return [int(value) for value in box]

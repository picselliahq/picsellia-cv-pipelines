import os

from picsellia_cv_engine.models.data.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from picsellia_cv_engine.models.model.picsellia_prediction import (
    PicselliaConfidence,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)
from picsellia_cv_engine.models.steps.model.predictor.model_context_predictor import (
    ModelContextPredictor,
)
from ultralytics.engine.results import Results

from pipelines.yolov8.training.pipeline_utils.model.ultralytics_model_context import (
    UltralyticsModelContext,
)


class UltralyticsSegmentationModelContextPredictor(
    ModelContextPredictor[UltralyticsModelContext]
):
    """
    A predictor class that handles model inference and result post-processing for segmentation tasks
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
        return [self._run_inference(batch) for batch in image_batches]

    def _run_inference(self, batch_paths: list[str]) -> Results:
        return self.model_context.loaded_model(batch_paths)

    def post_process_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[Results],
        dataset_context: TBaseDatasetContext,
    ) -> list[PicselliaPolygonPrediction]:
        return [
            prediction
            for batch_paths, batch_result in zip(
                image_batches, batch_results, strict=False
            )
            for prediction in self._post_process(
                batch_paths, batch_result, dataset_context
            )
        ]

    def _post_process(
        self,
        image_paths: list[str],
        batch_prediction: Results,
        dataset_context: TBaseDatasetContext,
    ) -> list[PicselliaPolygonPrediction]:
        processed_predictions = []

        for image_path, prediction in zip(image_paths, batch_prediction, strict=False):
            asset_id = os.path.basename(image_path).split(".")[0]
            asset = dataset_context.dataset_version.list_assets(ids=[asset_id])[0]

            polygons, labels, confidences = self.format_predictions(
                prediction=prediction, dataset_context=dataset_context
            )

            processed_predictions.append(
                PicselliaPolygonPrediction(
                    asset=asset,
                    polygons=polygons,
                    labels=labels,
                    confidences=confidences,
                )
            )

        return processed_predictions

    def format_predictions(
        self, prediction: Results, dataset_context: TBaseDatasetContext
    ):
        if prediction.masks is None:
            return [], [], []

        # Extract polygon segmentation masks
        polygons_list = [
            self.format_polygons(polygon) for polygon in prediction.masks.xy
        ]

        # Convert to Picsellia types
        picsellia_polygons = [PicselliaPolygon(points) for points in polygons_list]
        picsellia_labels = [
            self.get_picsellia_label(
                prediction.names[int(cls.cpu().numpy())],
                dataset_context=dataset_context,
            )
            for cls in prediction.boxes.cls
        ]
        picsellia_confidences = [
            PicselliaConfidence(float(conf.cpu().numpy()))
            for conf in prediction.boxes.conf
        ]

        return picsellia_polygons, picsellia_labels, picsellia_confidences

    @staticmethod
    def format_polygons(polygon):
        """
        Convert polygon mask to integer points.
        """
        return polygon.astype(int).tolist()

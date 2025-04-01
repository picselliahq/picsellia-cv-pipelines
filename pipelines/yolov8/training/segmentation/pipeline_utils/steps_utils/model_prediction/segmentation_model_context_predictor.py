import os

from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.models import (
    PicselliaConfidence,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)
from picsellia_cv_engine.services.base.model.predictor.model_predictor import (
    ModelPredictor,
)
from ultralytics.engine.results import Results

from pipelines.yolov8.training.pipeline_utils.model.ultralytics_model import (
    UltralyticsModel,
)


class UltralyticsSegmentationModelPredictor(ModelPredictor[UltralyticsModel]):
    """
    A predictor class that handles model inference and result post-processing for segmentation tasks
    using the Ultralytics framework.
    """

    def __init__(self, model: UltralyticsModel):
        super().__init__(model)

    def pre_process_dataset(self, dataset: TBaseDataset) -> list[str]:
        if not dataset.images_dir:
            raise ValueError("No images directory found in the dataset.")

        return [
            os.path.join(dataset.images_dir, image_name)
            for image_name in os.listdir(dataset.images_dir)
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
        return self.model.loaded_model(batch_paths)

    def post_process_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[Results],
        dataset: TBaseDataset,
    ) -> list[PicselliaPolygonPrediction]:
        return [
            prediction
            for batch_paths, batch_result in zip(
                image_batches, batch_results, strict=False
            )
            for prediction in self._post_process(batch_paths, batch_result, dataset)
        ]

    def _post_process(
        self,
        image_paths: list[str],
        batch_prediction: Results,
        dataset: TBaseDataset,
    ) -> list[PicselliaPolygonPrediction]:
        processed_predictions = []

        for image_path, prediction in zip(image_paths, batch_prediction, strict=False):
            asset_id = os.path.basename(image_path).split(".")[0]
            asset = dataset.dataset_version.list_assets(ids=[asset_id])[0]

            polygons, labels, confidences = self.format_predictions(
                prediction=prediction, dataset=dataset
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

    def format_predictions(self, prediction: Results, dataset: TBaseDataset):
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
                dataset=dataset,
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

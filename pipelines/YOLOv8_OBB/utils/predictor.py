import os

from ultralytics.engine.results import Results

from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.models.picsellia_prediction import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)
from picsellia_cv_engine.core.services.model.predictor.model_predictor import (
    ModelPredictor,
)
from picsellia_cv_engine.frameworks.ultralytics.model.model import UltralyticsModel


class UltralyticsObbModelPredictor(ModelPredictor[UltralyticsModel]):
    """Run inference with a YOLO-OBB model and convert each oriented bounding
    box into a 4-point Picsellia polygon prediction."""

    def run_inference_on_batches(self, image_batches: list[list[str]]) -> list[Results]:
        return [self.model.loaded_model(batch) for batch in image_batches]

    def post_process_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[Results],
        dataset: TBaseDataset,
    ) -> list[PicselliaPolygonPrediction]:
        predictions: list[PicselliaPolygonPrediction] = []
        for batch_paths, batch_result in zip(
            image_batches, batch_results, strict=False
        ):
            predictions.extend(
                self._post_process(
                    image_paths=batch_paths,
                    batch_prediction=batch_result,
                    dataset=dataset,
                )
            )
        return predictions

    def _post_process(
        self,
        image_paths: list[str],
        batch_prediction: Results,
        dataset: TBaseDataset,
    ) -> list[PicselliaPolygonPrediction]:
        processed: list[PicselliaPolygonPrediction] = []
        for image_path, prediction in zip(image_paths, batch_prediction, strict=False):
            asset_id = os.path.basename(image_path).split(".")[0]
            asset = dataset.dataset_version.list_assets(ids=[asset_id])[0]

            polygons, labels, confidences = self._format(
                prediction=prediction, dataset=dataset
            )

            processed.append(
                PicselliaPolygonPrediction(
                    asset=asset,
                    polygons=polygons,
                    labels=labels,
                    confidences=confidences,
                )
            )
        return processed

    def _format(
        self, prediction: Results, dataset: TBaseDataset
    ) -> tuple[list[PicselliaPolygon], list[PicselliaLabel], list[PicselliaConfidence]]:
        obb = prediction.obb
        if obb is None or len(obb) == 0:
            return [], [], []

        # Pixel-space 4-corner coordinates, shape (N, 4, 2).
        corners = obb.xyxyxyxy.cpu().numpy()
        polygons = [
            PicselliaPolygon(
                [[int(round(float(x))), int(round(float(y)))] for x, y in box]
            )
            for box in corners
        ]
        labels = [
            self.get_picsellia_label(
                category_name=prediction.names[int(cls)],
                dataset=dataset,
            )
            for cls in obb.cls.cpu().numpy()
        ]
        confidences = [
            PicselliaConfidence(float(conf)) for conf in obb.conf.cpu().numpy()
        ]

        return polygons, labels, confidences

import logging
import math

import numpy as np
import tqdm
from picsellia import Client
from picsellia.exceptions import (
    InsufficientResourcesError,
    PicselliaError,
    ResourceNotFoundError,
)
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.label import Label
from picsellia.types.enums import InferenceType

from pipelines.yolov8.pre_annotation.pipeline_utils.parameters.processing_yolov8_preannotation_parameters import (
    ProcessingYOLOV8PreannotationParameters,
)
from pipelines.yolov8.training.classification.pipeline_utils.model.ultralytics_model import (
    UltralyticsModel,
)


def _check_model_type_sanity(model_version) -> None:
    if model_version.type == InferenceType.NOT_CONFIGURED:
        raise PicselliaError(
            f"Can't run pre-annotation job, {model_version.name} type not configured."
        )


def _type_coherence_check(dataset_version, model_version):
    if dataset_version.type == InferenceType.NOT_CONFIGURED:
        dataset_version.set_type(model_version.type)
    elif dataset_version.type != model_version.type:
        raise PicselliaError(
            f"Can't run pre-annotation job on a {dataset_version.type} with {model_version.type} model."
        )
    return dataset_version


def _set_dataset_version_type(self) -> None:
    self.dataset_version.set_type(self.model_version.type)
    logging.info(
        f"Setting dataset {self.dataset_version.name}/{self.dataset_version.version} to type {self.model_version.type}"
    )


def _get_model_labels_name(model_version):
    model_infos = model_version.sync()

    if "labels" not in model_infos.keys():
        raise InsufficientResourcesError(
            f"Can't find labelmap for model {model_version.name}"
        )

    if not isinstance(model_infos["labels"], dict):
        raise InsufficientResourcesError(
            f"Invalid labelmap type, expected 'dict', got {type(model_infos['labels'])}"
        )

    model_labels = list(model_infos["labels"].values())
    return model_labels, model_infos


class CocoFormatter:
    """Helper class to manage COCO format operations"""

    def __init__(self, dataset_labels_name: list[str]):
        self.coco: dict[str, list] = {"images": [], "annotations": [], "categories": []}
        self.image_ids: set[int] = set()
        self.annotation_counter = 0
        self.image_counter = 0

        # Create COCO categories from dataset labels
        self.category_mapping = {}
        for idx, label_name in enumerate(dataset_labels_name, start=1):
            self.coco["categories"].append({"id": idx, "name": label_name})
            self.category_mapping[label_name] = idx

    def add_image(self, asset: Asset) -> int:
        """Add an image to the COCO format and return its ID"""
        self.image_counter += 1
        image_id = self.image_counter

        if image_id not in self.image_ids:
            self.coco["images"].append(
                {
                    "id": image_id,
                    "file_name": asset.id_with_extension,
                    "width": asset.width,
                    "height": asset.height,
                }
            )
            self.image_ids.add(image_id)

        return image_id

    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox: list[int],
        score: float,
        area: float,
        segmentation: list | None = None,
        is_classification: bool = False,
    ) -> int:
        """Add an annotation to the COCO format and return its ID"""
        self.annotation_counter += 1
        annotation_id = self.annotation_counter

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": float(score),
            "area": area,
            "iscrowd": 0,
        }

        if segmentation:
            annotation["segmentation"] = [segmentation]

        if is_classification:
            annotation["is_classification"] = True

        self.coco["annotations"].append(annotation)
        return annotation_id


class GeometryUtils:
    """Static utility methods for geometry operations"""

    @staticmethod
    def rescale_normalized_segment(
        segment: list, width: int, height: int
    ) -> list[list[int]]:
        """Convert normalized segments to pixel coordinates"""
        return [[int(box[0] * height), int(box[1] * width)] for box in segment]

    @staticmethod
    def format_polygons(predictions) -> list[list[tuple[int, int]]]:
        """Convert prediction masks to polygon format"""
        if predictions.masks is None:
            return []
        polygons = predictions.masks.xy
        casted_polygons = [polygon.astype(int) for polygon in polygons]
        return [polygon.tolist() for polygon in casted_polygons]

    @staticmethod
    def polygon_bbox(polygon: list[list[int]]) -> list[int]:
        """Calculate bounding box from polygon points"""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x_min = min(xs)
        y_min = min(ys)
        return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]

    @staticmethod
    def polygon_area(polygon: list[list[int]]) -> float:
        """Calculate area of a polygon using shoelace formula"""
        area = 0.0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0


class LabelManager:
    """Manage label operations and mappings"""

    def __init__(self, dataset_version: DatasetVersion):
        self.dataset_version = dataset_version
        self.labelmap = self._get_labelmap(dataset_version)
        self.dataset_labels_name = [
            label.name for label in dataset_version.list_labels()
        ]

    def _get_labelmap(self, dataset_version: DatasetVersion) -> dict[str, Label]:
        """Create a mapping of label names to Label objects"""
        return {label.name: label for label in dataset_version.list_labels()}

    def get_label_by_name(self, label_name: str) -> Label:
        """Get a Label object by name, raising an error if not found"""
        if label_name not in self.labelmap:
            raise ValueError(f"The label {label_name} does not exist in the labelmap.")
        return self.labelmap[label_name]

    def check_labels_coherence(self, model_labels_name: list[str]) -> bool:
        """Check if at least one model label exists in the dataset"""
        intersecting_labels = set(model_labels_name).intersection(
            self.dataset_labels_name
        )
        logging.info(
            f"Pre-annotation Job will process classes: {list(intersecting_labels)}"
        )
        return len(intersecting_labels) > 0

    def check_exact_label_match(self, model_labels_name: list[str]) -> None:
        """Verify that model and dataset labels match exactly"""
        if set(model_labels_name) != set(self.dataset_labels_name):
            raise ValueError(
                f"Model and dataset labels don't match exactly. "
                f"Model: {model_labels_name}, Dataset: {self.dataset_labels_name}. "
                "Solution: either add the labels manually in Picsellia, "
                "or change the strategy to 'add' to add model labels to the dataset."
            )

    def add_missing_labels(self, model_labels_name: list[str]) -> None:
        """Add model labels to dataset if they don't exist"""
        for label in tqdm.tqdm(model_labels_name):
            if label not in self.dataset_labels_name:
                self.dataset_version.create_label(name=label)

        # Update label information
        self.labelmap = self._get_labelmap(self.dataset_version)
        self.dataset_labels_name = [
            label.name for label in self.dataset_version.list_labels()
        ]
        logging.info(f"Labels created: {self.dataset_labels_name}")


class AnnotationProcessor:
    """Process predictions and create annotations"""

    def __init__(self, label_manager: LabelManager, coco_formatter: CocoFormatter):
        self.label_manager = label_manager
        self.coco_formatter = coco_formatter

    def process_detection(
        self, asset: Asset, prediction, confidence_threshold: float
    ) -> None:
        """Process object detection predictions"""
        boxes = prediction.boxes.xyxyn.cpu().numpy()
        scores = prediction.boxes.conf.cpu().numpy()
        labels = prediction.boxes.cls.cpu().numpy().astype(np.int16)

        # Limit the number of boxes to reduce processing time
        nb_box_limit = min(100, len(boxes))
        image_id = self.coco_formatter.add_image(asset)

        for i in range(nb_box_limit):
            if scores[i] >= confidence_threshold:
                try:
                    label_name = prediction.names[labels[i]]
                    # Verify label exists
                    self.label_manager.get_label_by_name(label_name)

                    # Calculate bounding box in pixels
                    e = boxes[i].tolist()
                    bbox = [
                        int(e[0] * asset.width),
                        int(e[1] * asset.height),
                        int((e[2] - e[0]) * asset.width),
                        int((e[3] - e[1]) * asset.height),
                    ]
                    area = bbox[2] * bbox[3]

                    self.coco_formatter.add_annotation(
                        image_id=image_id,
                        category_id=self.coco_formatter.category_mapping[label_name],
                        bbox=bbox,
                        score=float(scores[i]),
                        area=area,
                    )
                except ResourceNotFoundError as e:
                    logging.error(e)
                    continue

    def process_segmentation(
        self, asset: Asset, predictions, confidence_threshold: float
    ) -> None:
        """Process segmentation predictions"""
        scores = predictions.boxes.conf.cpu().numpy()
        labels = predictions.boxes.cls.cpu().numpy().astype(np.int16)
        masks = GeometryUtils.format_polygons(predictions=predictions)

        # Limit the number of polygons to reduce processing time
        nb_polygons_limit = min(100, len(masks))
        image_id = self.coco_formatter.add_image(asset)

        for i in range(nb_polygons_limit):
            if scores[i] >= confidence_threshold:
                try:
                    label_name = predictions.names[labels[i]]
                    self.label_manager.get_label_by_name(label_name)

                    segmentation = masks[i]
                    segmentation_list = [list(point) for point in segmentation]
                    bbox = GeometryUtils.polygon_bbox(segmentation_list)
                    area = GeometryUtils.polygon_area(segmentation_list)
                    flat_segmentation = [
                        coord for point in segmentation for coord in point
                    ]

                    self.coco_formatter.add_annotation(
                        image_id=image_id,
                        category_id=self.coco_formatter.category_mapping[label_name],
                        bbox=bbox,
                        score=float(scores[i]),
                        area=area,
                        segmentation=flat_segmentation,
                    )
                except ResourceNotFoundError as e:
                    logging.error(e)
                    continue

    def process_classification(
        self, asset: Asset, predictions, confidence_threshold: float
    ) -> None:
        """Process classification predictions"""
        image_id = self.coco_formatter.add_image(asset)

        if float(predictions.probs.top1conf.cpu().numpy()) >= confidence_threshold:
            label_name = predictions.names[int(predictions.probs.top1)]

            try:
                self.label_manager.get_label_by_name(label_name)

                # For classification, create an annotation covering the whole image
                bbox = [0, 0, asset.width, asset.height]
                area = asset.width * asset.height

                self.coco_formatter.add_annotation(
                    image_id=image_id,
                    category_id=self.coco_formatter.category_mapping[label_name],
                    bbox=bbox,
                    score=float(predictions.probs.top1conf.cpu().numpy()),
                    area=area,
                    is_classification=True,
                )

                logging.info(
                    f"Asset: {asset.filename} pre-annotated with label: {label_name}"
                )
            except ResourceNotFoundError as e:
                logging.error(e)


class PreAnnotator:
    """Main class for handling pre-annotation process"""

    def __init__(
        self,
        client: Client,
        dataset_version: DatasetVersion,
        model: UltralyticsModel,
        model_labels: list[str],
        parameters: ProcessingYOLOV8PreannotationParameters,
    ) -> None:
        self.client = client
        self.dataset_version = dataset_version
        self.model = model
        self.model_labels_name = model_labels
        self.parameters = parameters

        # Initialize components
        self.label_manager = LabelManager(dataset_version)
        self.coco_formatter = CocoFormatter(self.label_manager.dataset_labels_name)
        self.annotation_processor = AnnotationProcessor(
            self.label_manager, self.coco_formatter
        )

    def setup_preannotation_job(self) -> None:
        """Configure the pre-annotation job"""
        logging.info(
            f"Configuring pre-annotation task for dataset {self.dataset_version.name}/{self.dataset_version.version}"
        )

        # Handle label matching strategy
        if self.parameters.label_matching_strategy == "exact":
            self.label_manager.check_exact_label_match(self.model_labels_name)
        elif self.parameters.label_matching_strategy == "add":
            self.label_manager.add_missing_labels(self.model_labels_name)
            # Recreate the COCO formatter with updated labels
            self.coco_formatter = CocoFormatter(self.label_manager.dataset_labels_name)
            self.annotation_processor = AnnotationProcessor(
                self.label_manager, self.coco_formatter
            )
        else:
            raise ValueError(
                f"Unknown label matching strategy: {self.parameters.label_matching_strategy}"
            )

        # Verify label compatibility
        self.label_manager.check_labels_coherence(self.model_labels_name)

    def preannotate(self, confidence_threshold: float) -> dict:
        """Run pre-annotation on the dataset"""
        dataset_size = self.dataset_version.sync()["size"]
        batch_size = min(self.parameters.batch_size, dataset_size)
        image_size = self.parameters.image_size
        total_batch_number = math.ceil(dataset_size / batch_size)

        for batch_number in tqdm.tqdm(range(total_batch_number)):
            # Get batch of assets
            assets = self.dataset_version.list_assets(
                limit=batch_size, offset=batch_number * batch_size
            )
            url_list = [asset.sync()["data"]["presigned_url"] for asset in assets]

            # Get predictions for batch
            predictions = self.model.loaded_model(url_list, imgsz=image_size)

            # Process each asset and prediction
            for asset, prediction in list(zip(assets, predictions, strict=False)):
                if len(prediction) > 0:
                    self._process_prediction(asset, prediction, confidence_threshold)

        return self.coco_formatter.coco

    def _process_prediction(
        self, asset: Asset, prediction, confidence_threshold: float
    ) -> None:
        """Process prediction based on dataset type"""
        inference_type = self.dataset_version.type

        if inference_type == InferenceType.OBJECT_DETECTION:
            self.annotation_processor.process_detection(
                asset, prediction, confidence_threshold
            )
        elif inference_type == InferenceType.SEGMENTATION:
            self.annotation_processor.process_segmentation(
                asset, prediction, confidence_threshold
            )
        elif inference_type == InferenceType.CLASSIFICATION:
            self.annotation_processor.process_classification(
                asset, prediction, confidence_threshold
            )

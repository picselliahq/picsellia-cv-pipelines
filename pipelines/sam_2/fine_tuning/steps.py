import os
from collections.abc import Callable

import numpy as np
from picsellia_cv_engine import Pipeline
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.data import TBaseDataset
from picsellia_cv_engine.core.models import (
    PicselliaConfidence,
    PicselliaLabel,
    PicselliaPolygon,
    PicselliaPolygonPrediction,
)
from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.frameworks.sam2.model.model import SAM2Model
from picsellia_cv_engine.frameworks.sam2.services.predictor import SAM2ModelPredictor
from picsellia_cv_engine.steps.base.model.evaluator import evaluate_model_impl
from PIL import Image


def _get_or_load_predictor(model: SAM2Model, device: str) -> SAM2ModelPredictor:
    try:
        predictor = model.loaded_predictor
    except ValueError as e:
        if not model.trained_weights_path or not model.config_path:
            raise ValueError(
                "SAM2Model requires both trained_weights_path and config_path."
            ) from e
        _, predictor = model.load_weights(
            weights_path=model.trained_weights_path,
            config_path=model.config_path,
            device=device,
        )
        model.set_loaded_predictor(predictor)
    return SAM2ModelPredictor(predictor=predictor)


def _build_label_resolver(dataset: TBaseDataset) -> Callable[[int], PicselliaLabel]:
    categories = dataset.coco_data.get("categories", [])
    catid_to_name = {
        c["id"]: c["name"] for c in categories if "id" in c and "name" in c
    }
    fallback = next(iter(dataset.labelmap.values())) if dataset.labelmap else "unknown"

    def resolve(cat_id: int) -> PicselliaLabel:
        name = catid_to_name.get(cat_id)
        if name and name in dataset.labelmap:
            return PicselliaLabel(dataset.labelmap[name])
        return PicselliaLabel(fallback)

    return resolve


def _index_annotations_by_image(dataset: TBaseDataset) -> dict[int, list[dict]]:
    ann_by_image: dict[int, list[dict]] = {}
    for ann in dataset.coco_data.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    return ann_by_image


def _predict_for_image(
    img_path: str,
    img_id: int,
    predictor: SAM2ModelPredictor,
    ann_by_image: dict[int, list[dict]],
    resolve_label: Callable[[int], PicselliaLabel],
) -> tuple[list[PicselliaPolygon], list[PicselliaLabel], list[PicselliaConfidence]]:
    polygons: list[PicselliaPolygon] = []
    labels: list[PicselliaLabel] = []
    confs: list[PicselliaConfidence] = []

    if not os.path.exists(img_path):
        return polygons, labels, confs

    img_np = np.array(Image.open(img_path).convert("RGB"))
    predictor.preprocess(image=img_np)

    for ann in ann_by_image.get(img_id, []):
        x, y, w, h = ann["bbox"][:4]
        box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

        raw = predictor.run_inference(box=box_xyxy, multimask_output=False)
        polys = predictor.post_process(raw)
        if not polys:
            continue

        poly = polys[0]["polygon"]
        score = float(polys[0]["score"])
        if not poly:
            continue

        polygons.append(PicselliaPolygon(points=poly))
        labels.append(resolve_label(ann.get("category_id", 0)))
        confs.append(PicselliaConfidence(value=score))

    return polygons, labels, confs


@step
def evaluate_sam2_model(model: SAM2Model, dataset: TBaseDataset) -> None:
    """
    Evaluate SAM2 using COCO ground-truth bounding boxes as prompts:
    - For each image, use each COCO bbox as a prompt box.
    - Run SAM2, convert masks to polygons, attach labels/confidences.
    - Submit predictions to the generic evaluator.
    """
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    device = getattr(context.hyperparameters, "device", "cuda")

    model_predictor = _get_or_load_predictor(model=model, device=device)
    resolve_label = _build_label_resolver(dataset)
    ann_by_image = _index_annotations_by_image(dataset)
    filename_to_asset = {a.filename: a for a in dataset.assets}

    predictions: list[PicselliaPolygonPrediction] = []
    for img_info in dataset.coco_data.get("images", []):
        fname = img_info["file_name"]
        asset = filename_to_asset.get(fname)
        if asset is None:
            continue

        img_path = os.path.join(dataset.images_dir, fname)
        polys, labels, confs = _predict_for_image(
            img_path=img_path,
            img_id=img_info["id"],
            predictor=model_predictor,
            ann_by_image=ann_by_image,
            resolve_label=resolve_label,
        )

        if polys:
            predictions.append(
                PicselliaPolygonPrediction(
                    asset=asset, polygons=polys, labels=labels, confidences=confs
                )
            )

    evaluate_model_impl(
        context=context,
        picsellia_predictions=predictions,
        inference_type=model.model_version.type,
        assets=dataset.assets,
        output_dir=os.path.join(model.results_dir, "inference"),
        training_labelmap=context.experiment.get_log("labelmap").data,
    )

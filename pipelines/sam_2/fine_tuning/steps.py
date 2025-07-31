import os
import random
import shutil
import subprocess
import sys

import numpy as np
import supervision as sv
import torch
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from utils.sam2_utils import (
    convert_coco_to_png_masks,
    load_coco_annotations,
    normalize_filenames,
    parse_and_log_sam2_output,
    prepare_directories,
)


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sam2_root = os.path.join(current_dir, "sam2")
    sys.path.insert(0, sam2_root)

    img_root = os.path.join(sam2_root, "data", "JPEGImages")
    ann_root = os.path.join(sam2_root, "data", "Annotations")
    prepare_directories(img_root, ann_root)

    # --- Copier fichiers nécessaires ---
    source_images = picsellia_datasets["train"].images_dir
    source_annotations = picsellia_datasets["train"].annotations_dir
    coco_file = next(f for f in os.listdir(source_annotations) if f.endswith(".json"))
    coco_path = os.path.join(source_annotations, coco_file)
    shutil.copy(coco_path, os.path.join(context.working_dir, "coco_annotations.json"))
    shutil.copy(
        picsellia_model.pretrained_weights_path, os.path.join(sam2_root, "checkpoints")
    )
    pretrained_weights_name = os.path.basename(picsellia_model.pretrained_weights_path)
    picsellia_model.pretrained_weights_path = os.path.join(
        sam2_root, "checkpoints", pretrained_weights_name
    )

    # --- Masques PNG ---
    coco = load_coco_annotations(
        os.path.join(context.working_dir, "coco_annotations.json")
    )
    convert_coco_to_png_masks(coco, source_images, img_root, ann_root)
    normalize_filenames([img_root, ann_root])

    # --- Lancer entraînement ---
    experiment_log_dir = os.path.join(picsellia_model.results_dir, "sam2_logs")
    os.makedirs(experiment_log_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [sam2_root, os.path.join(sam2_root, "training")]
    )

    log_file = os.path.join(experiment_log_dir, "train_stdout.log")
    overrides = [
        f"scratch.train_batch_size={context.hyperparameters.batch_size}",
        f"scratch.resolution={context.hyperparameters.image_size}",
        f"scratch.base_lr={context.hyperparameters.learning_rate}",
        f"scratch.num_epochs={context.hyperparameters.epochs}",
        f"trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path={os.path.join('checkpoints', pretrained_weights_name)}",
    ]

    picsellia_model.results_dir = os.path.join(
        sam2_root, "sam2_logs", "configs", "train.yaml"
    )

    command = [
        sys.executable,
        "-m",
        "training.train",
        "-c",
        "configs/train.yaml",
        "--use-cluster",
        "0",
        "--num-gpus",
        "1",
        *overrides,
    ]

    process = subprocess.Popen(
        command,
        cwd=sam2_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    parse_and_log_sam2_output(process, context, log_file)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

    # --- Sauvegarde checkpoint ---
    checkpoint_path = os.path.join(
        picsellia_model.results_dir, "checkpoints", "checkpoint.pt"
    )
    if os.path.exists(checkpoint_path):
        picsellia_model.save_artifact_to_experiment(
            experiment=context.experiment,
            artifact_name="model-latest",
            artifact_path=checkpoint_path,
        )
        picsellia_model.trained_weights_path = checkpoint_path
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


@step()
def evaluate(picsellia_model: Model, dataset: CocoDataset):
    from picsellia_cv_engine.core.models.picsellia_prediction import (
        PicselliaConfidence,
        PicselliaLabel,
        PicselliaPolygon,
        PicselliaPolygonPrediction,
    )
    from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl

    context = Pipeline.get_active_context()
    eval_dir = os.path.join(picsellia_model.results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # === Chargement des modèles ===
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    fine_model = build_sam2(
        model_cfg, picsellia_model.trained_weights_path, device="cuda"
    )
    base_model = build_sam2(
        model_cfg, picsellia_model.pretrained_weights_path, device="cuda"
    )

    fine_mask_gen = SAM2AutomaticMaskGenerator(fine_model)
    base_mask_gen = SAM2AutomaticMaskGenerator(base_model)

    # === Liste des images ===
    all_images = [
        os.path.join(dataset.images_dir, f)
        for f in os.listdir(dataset.images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # === a) Visualisation de quelques comparaisons ===
    sampled_images = random.sample(all_images, min(5, len(all_images)))
    for i, img_path in enumerate(sampled_images):
        image = np.array(Image.open(img_path).convert("RGB"))

        fine_result = fine_mask_gen.generate(image)
        fine_detections = sv.Detections.from_sam(sam_result=fine_result)
        fine_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        fine_image = fine_annotator.annotate(image.copy(), detections=fine_detections)

        base_result = base_mask_gen.generate(image)
        base_detections = sv.Detections.from_sam(sam_result=base_result)
        base_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        base_image = base_annotator.annotate(image.copy(), detections=base_detections)

        concat = Image.fromarray(np.hstack((fine_image, base_image)))
        output_path = os.path.join(eval_dir, f"comparison_{i}.png")
        concat.save(output_path)

        context.experiment.log(
            name=f"eval/inference_comparison_{i}",
            data=output_path,
            type="image",
        )

    first_label_name = list(dataset.labelmap.keys())[0]
    picsellia_first_label = dataset.labelmap[first_label_name]

    # === b) Évaluation complète sur tout le dataset ===
    predictions: list[PicselliaPolygonPrediction] = []

    for img_path in all_images:
        image = np.array(Image.open(img_path).convert("RGB"))
        filename = os.path.basename(img_path)
        asset_id = os.path.splitext(filename)[0]

        # Get corresponding asset
        asset = dataset.dataset_version.list_assets(ids=[asset_id])[0]

        # Inference
        fine_result = fine_mask_gen.generate(image)

        asset_polygons = []
        asset_labels = []
        asset_confidences = []

        for mask_dict in fine_result:
            mask = mask_dict.get("segmentation", None)
            if mask is None:
                continue

            polygons = sv.mask_to_polygons(mask.astype(np.uint8))

            for polygon in polygons:
                if len(polygon) == 0:
                    continue

                polygon_points = [[int(x), int(y)] for x, y in polygon]
                asset_polygons.append(PicselliaPolygon(polygon_points))
                asset_labels.append(PicselliaLabel(picsellia_first_label))
                asset_confidences.append(PicselliaConfidence(1.0))

        # Append one prediction per asset
        if asset_polygons:
            predictions.append(
                PicselliaPolygonPrediction(
                    asset=asset,
                    polygons=asset_polygons,
                    labels=asset_labels,
                    confidences=asset_confidences,
                )
            )

    # === Upload evaluation to experiment ===
    evaluate_model_impl(
        context=context,
        picsellia_predictions=predictions,
        inference_type=picsellia_model.model_version.type,
        assets=dataset.assets,
        output_dir=os.path.join(context.working_dir, "evaluation"),
        training_labelmap=context.experiment.get_log("labelmap").data,
    )

    print("✅ Évaluation complète SAM2 terminée et logguée")

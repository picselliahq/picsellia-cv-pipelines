import json
import os
import shutil
import subprocess
import sys
from typing import Any

import yaml
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from PIL import Image, ImageDraw


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()

    # --- Dossiers de travail SAM2 ---
    img_dir = os.path.join(context.working_dir, "JPEGImages")
    ann_dir = os.path.join(context.working_dir, "Annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # --- Copier les images et annotations depuis Picsellia ---
    source_images = picsellia_datasets["train"].images_dir
    source_annotations = picsellia_datasets["train"].annotations_dir
    for f in os.listdir(source_images):
        shutil.copy(os.path.join(source_images, f), os.path.join(img_dir, f))
    coco_file = [f for f in os.listdir(source_annotations) if f.endswith(".json")][0]
    coco_path = os.path.join(source_annotations, coco_file)
    shutil.copy(coco_path, os.path.join(context.working_dir, "coco_annotations.json"))

    # --- Générer les masques PNG à partir des polygones COCO ---
    with open(os.path.join(context.working_dir, "coco_annotations.json")) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    annotations_by_image: dict[str, Any] = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id, annotations in annotations_by_image.items():
        img_info = images_by_id[img_id]
        width, height = img_info["width"], img_info["height"]
        file_name = os.path.splitext(img_info["file_name"])[0]

        mask = Image.new("I", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for ann in annotations:
            if ann.get("iscrowd", 0) == 1 or "segmentation" not in ann:
                continue
            for seg in ann["segmentation"]:
                if len(seg) >= 6:
                    poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                    draw.polygon(poly, fill=ann["category_id"])

        mask = mask.convert("L")
        mask.save(os.path.join(ann_dir, f"{file_name}.png"))

    # --- Charger et modifier le template YAML SAM2 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sam2_root = os.path.join(current_dir, "sam2")
    config_template_path = os.path.join(
        sam2_root,
        "sam2",
        "configs",
        "sam2.1_training",
        "sam2.1_hiera_b+_MOSE_finetune.yaml",
    )
    config_output_path = os.path.join(sam2_root, "sam2", "sam2_config.yaml")
    experiment_log_dir = os.path.join(picsellia_model.results_dir, "sam2_logs")
    os.makedirs(experiment_log_dir, exist_ok=True)

    with open(config_template_path) as f:
        config = yaml.safe_load(f)

    config["dataset"]["img_folder"] = img_dir
    config["dataset"]["gt_folder"] = ann_dir
    config["launcher"]["experiment_log_dir"] = experiment_log_dir
    config["trainer"]["model"]["checkpoint"] = picsellia_model.pretrained_weights_path

    with open(config_output_path, "w") as f:
        yaml.safe_dump(config, f)

    # --- Lancer le training SAM2 ---
    sam2_train_script = os.path.join(sam2_root, "training", "train.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = sam2_root

    subprocess.run(
        [
            sys.executable,
            sam2_train_script,
            "-c",
            "sam2_config.yaml",
            "--use-cluster",
            "0",
            "--num-gpus",
            "1",
        ],
        check=True,
        env=env,
    )

    # --- Sauvegarder le checkpoint dans Picsellia ---
    checkpoint_path = os.path.join(experiment_log_dir, "checkpoints", "sam.pt")
    if os.path.exists(checkpoint_path):
        picsellia_model.save_artifact_to_experiment(
            experiment=context.experiment,
            artifact_name="finetuned-sam2",
            artifact_path=checkpoint_path,
        )
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

import os
import shutil
import subprocess
import sys

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from utils.sam2_utils import (
    convert_coco_to_png_masks,
    load_coco_annotations,
    normalize_filenames,
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

    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.train",
            "-c",
            "configs/train.yaml",
            "--use-cluster",
            "0",
            "--num-gpus",
            "1",
        ],
        cwd=sam2_root,
        env=env,
        check=True,
    )

    # --- Sauvegarde checkpoint ---
    checkpoint_path = os.path.join(experiment_log_dir, "checkpoints", "sam.pt")
    if os.path.exists(checkpoint_path):
        picsellia_model.save_artifact_to_experiment(
            experiment=context.experiment,
            artifact_name="finetuned-sam2",
            artifact_path=checkpoint_path,
        )
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

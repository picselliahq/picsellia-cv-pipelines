import json
import os
import subprocess
from pathlib import Path

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from tqdm import tqdm


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()

    root_dir = Path(picsellia_datasets.dataset_path)
    odvg_dir = root_dir / "odvg_data"
    os.makedirs(odvg_dir, exist_ok=True)

    train_dir = val_dir = None
    train_jsonl = val_json = None
    label_map_path = odvg_dir / "label_map.json"

    # Générer les fichiers ODVG + label map
    for dataset in picsellia_datasets:
        if dataset.name.lower() not in {"train", "val"}:
            continue
        name = dataset.name.lower()
        dataset_dir = odvg_dir / name
        os.makedirs(dataset_dir / "images", exist_ok=True)
        os.makedirs(dataset_dir / "annotations", exist_ok=True)

        dataset.download_assets(destination_dir=str(dataset_dir / "images"))
        dataset.download_annotations(destination_dir=str(dataset_dir / "annotations"))

        coco_file = os.path.join(dataset.annotations_dir, "coco_annotations.json")
        jsonl_file = os.path.join(dataset_dir, f"{name}.odvg.jsonl")

        if name == "train":
            train_dir = dataset_dir / "images"
            train_jsonl = jsonl_file
            coco_to_odvg(
                coco_path=coco_file,
                images_dir=train_dir,
                output_path=jsonl_file,
                label_map_path=label_map_path,
            )

        elif name == "val":
            val_dir = dataset_dir / "images"
            val_json = coco_file  # COCO val reste en COCO

    # Générer le fichier de config ODVG pour Open-GroundingDino
    generate_odvg_config(
        root_dir=root_dir,
        train_root=str(train_dir),
        jsonl_path=str(train_jsonl),
        label_map_path=str(label_map_path),
        val_root=str(val_dir),
        val_ann=str(val_json),
    )

    repo_path = Path(__file__).parent / "Open-GroundingDino"

    subprocess.run(
        [
            "bash",
            str(repo_path / "train_dist.sh"),
            "1",
            str(repo_path / "config/cfg_odvg.py"),
            str(root_dir / "datasets_mixed_odvg.json"),
            str(root_dir / "grounding_dino_logs"),
        ],
        check=True,
        cwd=str(repo_path),
    )

    # Sauvegarde du modèle (à adapter selon où est le .pth final)
    trained_model_path = (
        root_dir / "grounding_dino_logs" / "GroundingDINO" / "checkpoint.pth"
    )
    if trained_model_path.exists():
        picsellia_model.save_artifact_to_experiment(
            experiment=context.experiment,
            artifact_name="groundingdino-ckpt",
            artifact_path=str(trained_model_path),
        )


def coco_to_odvg(coco_path, images_dir, output_path, label_map_path):
    with open(coco_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    label_map = {str(i): cat for i, cat in enumerate(sorted(set(categories.values())))}
    category_to_id = {v: int(k) for k, v in label_map.items()}

    with open(label_map_path, "w") as lm_file:
        json.dump(label_map, lm_file, indent=2)

    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    with open(output_path, "w") as out_file:
        for img_id, anns in tqdm(annotations_by_image.items()):
            img = images[img_id]
            detection = {"instances": []}
            for ann in anns:
                category = categories[ann["category_id"]]
                detection["instances"].append(
                    {
                        "bbox": ann["bbox"],
                        "label": category_to_id[category],
                        "category": category,
                    }
                )

            odvg_item = {
                "filename": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "detection": detection,
            }
            out_file.write(json.dumps(odvg_item) + "\n")


def generate_odvg_config(
    root_dir, train_root, jsonl_path, label_map_path, val_root, val_ann
):
    config = {
        "train": [
            {
                "root": train_root,
                "anno": jsonl_path,
                "label_map": label_map_path,
                "dataset_mode": "odvg",
            }
        ],
        "val": [
            {
                "root": val_root,
                "anno": val_ann,
                "label_map": None,
                "dataset_mode": "coco",
            }
        ],
    }
    config_path = root_dir / "datasets_mixed_odvg.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

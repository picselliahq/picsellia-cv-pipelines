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
    train_jsonl = val_jsonl = None
    label_map_path = odvg_dir / "label_map.json"

    for dataset in picsellia_datasets:
        if dataset.name.lower() not in {"train", "val"}:
            continue
        name = dataset.name.lower()
        coco_file = os.path.join(dataset.annotations_dir, "coco_annotations.json")
        jsonl_file = odvg_dir / name / f"{name}.odvg.jsonl"
        jsonl_file.parent.mkdir(parents=True, exist_ok=True)

        if name == "train":
            train_dir = dataset.images_dir
            train_jsonl = jsonl_file
            coco_to_odvg(
                coco_path=coco_file,
                images_dir=train_dir,
                output_path=jsonl_file,
                label_map_path=label_map_path,
            )

        elif name == "val":
            val_dir = dataset.images_dir
            val_jsonl = jsonl_file
            coco_to_odvg(
                coco_path=coco_file,
                images_dir=val_dir,
                output_path=jsonl_file,
                label_map_path=label_map_path,
            )

    generate_odvg_config(
        root_dir=root_dir,
        train_root=str(train_dir),
        jsonl_path=str(train_jsonl),
        label_map_path=str(label_map_path),
        val_root=str(val_dir),
        val_jsonl=str(val_jsonl),
    )

    repo_path = Path(__file__).parent / "Open-GroundingDino"

    py_config_path = repo_path / "config/cfg_odvg.py"
    append_odvg_eval_config(config_path=py_config_path, label_map_path=label_map_path)

    pipeline_root = Path(__file__).parent
    train_script = pipeline_root / "train.sh"

    subprocess.run(
        [
            "bash",
            train_script,
            "1",
            str(repo_path / "config/cfg_odvg.py"),
            str(root_dir / "datasets_mixed_odvg.json"),
            str(root_dir / "grounding_dino_logs"),
            picsellia_model.pretrained_weights_path,
        ],
        check=True,
        cwd=str(repo_path),
    )

    # Chemin final du checkpoint
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

    category_ids = sorted(categories.keys())
    coco_id_to_index = {coco_id: i for i, coco_id in enumerate(category_ids)}
    index_to_name = {i: categories[coco_id] for coco_id, i in coco_id_to_index.items()}

    with open(label_map_path, "w") as lm_file:
        json.dump(index_to_name, lm_file, indent=2)

    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    with open(output_path, "w") as out_file:
        for img_id, anns in tqdm(annotations_by_image.items()):
            img = images[img_id]
            detection = {"instances": []}
            for ann in anns:
                coco_cat_id = ann["category_id"]
                index_id = coco_id_to_index[coco_cat_id]
                category_name = categories[coco_cat_id]

                detection["instances"].append(
                    {
                        "bbox": ann["bbox"],
                        "label": index_id,
                        "category": category_name,
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
    root_dir, train_root, jsonl_path, label_map_path, val_root, val_jsonl
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
                "anno": val_jsonl,
                "label_map": label_map_path,
                "dataset_mode": "odvg",
            }
        ],
    }
    config_path = root_dir / "datasets_mixed_odvg.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def append_odvg_eval_config(config_path: Path, label_map_path: Path):
    with open(label_map_path) as f:
        label_map = json.load(f)

    label_list = [label_map[str(i)] for i in range(len(label_map))]

    with open(config_path, "a") as f:  # append mode
        f.write("\n\n# Auto-appended for ODVG training\n")
        f.write("use_coco_eval = False\n")
        f.write(f"label_list = {label_list}\n")

import json
import os
import subprocess
import sys

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()

    # === Utilise le contexte local existant ===
    train_dataset = picsellia_datasets["train"]
    working_dir = context.working_dir

    json_dir = os.path.join(working_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

    # === Préparation du fichier Hugging Face train.json ===
    train_json_path = os.path.join(json_dir, "train.json")
    export_picsellia_to_clip_json(train_dataset, train_json_path)

    # === Configuration du dossier de sortie ===
    output_dir = os.path.join(picsellia_model.results_dir, "clip_finetuned")
    os.makedirs(output_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_path = os.path.join(script_dir, "run_clip.py")

    # === Appel de run_clip.py avec les bons arguments ===
    command = [
        sys.executable,
        run_path,
        "--output_dir",
        output_dir,
        "--model_name_or_path",
        "openai/clip-vit-large-patch14-336",
        "--train_file",
        train_json_path,
        "--do_train",
        "--image_column",
        "image",
        "--caption_column",
        "caption",
        "--remove_unused_columns",
        "False",
        "--max_seq_length",
        "77",
        "--per_device_train_batch_size",
        str(context.hyperparameters.batch_size),
        "--num_train_epochs",
        str(context.hyperparameters.epochs),
        "--learning_rate",
        "5e-5",
        "--warmup_steps",
        "0",
        "--weight_decay",
        "0.1",
        "--overwrite_output_dir",
    ]

    subprocess.run(command, check=True)

    # === Sauvegarde du modèle fine-tuné dans l'expérience ===
    picsellia_model.save_artifact_to_experiment(
        experiment=context.experiment,
        artifact_name="clip-model",
        artifact_path=output_dir,
    )


def export_picsellia_to_clip_json(dataset: CocoDataset, output_path: str):
    coco = dataset.coco_data
    images_dir = dataset.images_dir

    # map category_id -> category_name
    id_to_category_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # map image_id -> caption
    id_to_caption = {
        ann["image_id"]: f"a photo of a {id_to_category_name[ann['category_id']]}"
        for ann in coco["annotations"]
    }

    # enrich each image dict with image_path and caption
    enriched_images = []
    for img in coco["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        image_path = os.path.join(images_dir, file_name)
        caption = id_to_caption.get(img_id)

        if caption:
            enriched = img.copy()
            enriched["image"] = image_path
            enriched["caption"] = caption
            enriched_images.append(enriched)

    with open(output_path, "w") as f:
        for item in enriched_images:
            json_line = json.dumps(item, separators=(",", ":"))
            f.write(json_line + "\n")

import glob
import json
import os
import re
import subprocess
import sys

import torch
from picsellia import Experiment
from picsellia.types.enums import LogType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from picsellia_cv_engine.core.contexts.training.local_training_context import (
    LocalTrainingContext,
)
from picsellia_cv_engine.core.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
from utils.evaluation import (
    apply_dbscan_clustering,
    generate_embeddings,
    load_stored_embeddings,
    reduce_dimensionality_umap,
    save_cluster_images_plot,
    save_clustering_plots,
    save_outliers_images,
)

TRAIN_METRICS = ["loss", "grad_norm", "learning_rate"]
EVAL_METRICS = ["eval_loss"]


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    """Main training step for the CLIP fine-tuning pipeline."""
    context = Pipeline.get_active_context()
    working_dir = context.working_dir
    os.makedirs(json_dir := os.path.join(working_dir, "json"), exist_ok=True)

    train_json = os.path.join(json_dir, "train.json")
    val_json = os.path.join(json_dir, "val.json")
    test_json = os.path.join(json_dir, "test.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = prepare_caption_model(device)

    for split_name, output_path in zip(
        ["train", "val", "test"], [train_json, val_json, test_json], strict=False
    ):
        export_dataset_to_clip_json(
            model=model,
            processor=processor,
            dataset=picsellia_datasets[split_name],
            output_path=output_path,
            device=device,
            prompt=context.hyperparameters.caption_prompt,
        )

    del model
    torch.cuda.empty_cache()

    output_dir = os.path.join(picsellia_model.results_dir, "clip_finetuned")
    os.makedirs(output_dir, exist_ok=True)
    run_script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_clip.py"
    )

    run_clip_training(
        model_name_or_path=picsellia_model.pretrained_weights_path,
        run_script_path=run_script_path,
        output_dir=output_dir,
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        context=context,
    )

    save_best_checkpoint(
        output_dir=output_dir,
        picsellia_model=picsellia_model,
        experiment=context.experiment,
    )


@step()
def evaluate_clip_embeddings(picsellia_model: Model, dataset: CocoDataset):
    """
    Step d’évaluation CLIP : embeddings, UMAP, clustering DBSCAN, logging des visualisations.
    """
    context = Pipeline.get_active_context()
    experiment: Experiment = context.experiment

    # Génération ou chargement des embeddings
    evaluation_results = os.path.join(picsellia_model.results_dir, "evaluation")
    embeddings_file = os.path.join(evaluation_results, "embeddings.npz")
    os.makedirs(evaluation_results, exist_ok=True)
    generate_embeddings(
        model_path=picsellia_model.exported_weights_path,
        images_dir=dataset.images_dir,
        embeddings_file_path=embeddings_file,
    )
    embeddings, image_paths = load_stored_embeddings(file_path=embeddings_file)

    # UMAP
    reduced = reduce_dimensionality_umap(embeddings=embeddings, n_components=2)

    # DBSCAN clustering
    eps = 0.3
    min_samples = 5
    labels = apply_dbscan_clustering(
        reduced, dbscan_eps=eps, dbscan_min_samples=min_samples
    )

    # Logging
    cluster_dir = os.path.join(evaluation_results, f"clusters_eps{eps}")
    os.makedirs(cluster_dir, exist_ok=True)

    save_clustering_plots(reduced, labels, results_dir=cluster_dir)
    save_cluster_images_plot(image_paths, labels, results_dir=cluster_dir)
    save_outliers_images(image_paths, labels, results_dir=cluster_dir)

    for filename in os.listdir(cluster_dir):
        if filename.endswith(".png"):
            experiment.log(
                name=f"clip-eval/{filename}",
                data=os.path.join(cluster_dir, filename),
                type=LogType.IMAGE,
            )

    print("✅ CLIP evaluation done, plots logged under 'clip-eval/*'")


def generate_caption(
    model, processor, image_path: str, prompt: str, device: str
) -> str:
    """Generates a caption for a given image using the BLIP model."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {image_path}") from e

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Remove incomplete final sentence if missing punctuation
    if not caption.strip().endswith("."):
        sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
        if len(sentences) > 1:
            caption = " ".join(sentences[:-1])
        else:
            caption = ""

    return caption


def export_dataset_to_clip_json(
    model, processor, dataset: CocoDataset, output_path: str, device: str, prompt: str
) -> None:
    """Exports a CocoDataset to a JSON file in a CLIP-compatible format with generated captions."""
    coco = dataset.coco_data
    images_dir = dataset.images_dir
    enriched_images: list[dict] = []

    for img in coco["images"]:
        image_path = os.path.join(images_dir, img["file_name"])
        caption = generate_caption(model, processor, image_path, prompt, device)
        enriched_images.append(
            {
                "image": image_path,
                "caption": caption,
                **img,
            }
        )

    with open(output_path, "w") as f:
        for item in enriched_images:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")


def prepare_caption_model(device: str):
    """Loads the BLIP model and processor onto the specified device."""
    processor = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl"
    )
    model = (
        InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl", device_map="auto"
        )
        .eval()
        .to(device)
    )
    return model, processor


def build_clip_command(
    model_name_or_path: str,
    script_path: str,
    output_dir: str,
    train_file: str,
    val_file: str,
    test_file: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
) -> list[str]:
    """Builds the CLI command to run the CLIP training script."""
    return [
        sys.executable,
        script_path,
        "--output_dir",
        output_dir,
        "--model_name_or_path",
        model_name_or_path,
        "--do_train",
        "--do_eval",
        "--do_predict",
        "--train_file",
        train_file,
        "--validation_file",
        val_file,
        "--test_file",
        test_file,
        "--image_column",
        "image",
        "--caption_column",
        "caption",
        "--remove_unused_columns",
        "False",
        "--max_seq_length",
        "77",
        "--per_device_train_batch_size",
        str(batch_size),
        "--num_train_epochs",
        str(epochs),
        "--learning_rate",
        str(learning_rate),
        "--warmup_steps",
        str(warmup_steps),
        "--weight_decay",
        str(weight_decay),
        "--overwrite_output_dir",
        "--logging_strategy",
        "epoch",
        "--eval_strategy",
        "epoch",
        "--save_strategy",
        "best",
        "--metric_for_best_model",
        "loss",
    ]


def parse_and_log_training_output(process, context, log_file_path):
    """Parses stdout from the CLIP training process and logs metrics to the experiment."""
    train_pattern = re.compile(
        r"\{.*?'loss':\s*([\d.eE+-]+),\s*'grad_norm':\s*([\d.eE+-]+),"
        r"\s*'learning_rate':\s*([\d.eE+-]+),\s*'epoch':\s*([\d.]+).*?\}"
    )
    metrics_pattern = re.compile(r"'(\w+)'[\s]*:[\s]*([\d.eE+-]+)")

    with open(log_file_path, "w") as log_file:
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

            # Step-wise training metrics
            match = train_pattern.search(line)
            if match:
                loss, grad_norm, lr, epoch = map(float, match.groups())
                context.experiment.log("train/loss", loss, LogType.LINE)
                context.experiment.log("train/grad_norm", grad_norm, LogType.LINE)
                context.experiment.log("train/learning_rate", lr, LogType.LINE)

            # Step-wise evaluation metrics
            elif "'eval_loss'" in line and "'epoch'" in line:
                metrics = dict(metrics_pattern.findall(line))
                if "eval_loss" in metrics:
                    context.experiment.log(
                        "val/loss",
                        float(metrics["eval_loss"]),
                        LogType.LINE,
                    )

            # Final metrics
            elif (
                "***** train metrics *****" in line
                or "***** eval metrics *****" in line
            ):
                prefix = "train" if "train" in line else "val"
                for _ in range(10):
                    metric_line = process.stdout.readline()
                    print(metric_line, end="")
                    log_file.write(metric_line)
                    for name, value in metrics_pattern.findall(metric_line):
                        log_name = (
                            f"{prefix}/loss"
                            if name == "eval_loss"
                            else f"{prefix}/{name}"
                        )
                        context.experiment.log(log_name, float(value), LogType.VALUE)


def run_clip_training(
    model_name_or_path: str,
    run_script_path: str,
    output_dir: str,
    train_json: str,
    val_json: str,
    test_json: str,
    context: PicselliaTrainingContext | LocalTrainingContext,
) -> None:
    """Executes CLIP training script and logs results."""

    command = build_clip_command(
        model_name_or_path=model_name_or_path,
        script_path=run_script_path,
        output_dir=output_dir,
        train_file=train_json,
        val_file=val_json,
        test_file=test_json,
        epochs=context.hyperparameters.epochs,
        batch_size=context.hyperparameters.batch_size,
        learning_rate=context.hyperparameters.learning_rate,
        warmup_steps=context.hyperparameters.warmup_steps,
        weight_decay=context.hyperparameters.weight_decay,
    )

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log_file_path = os.path.join(output_dir, "training_stdout.log")
    parse_and_log_training_output(process, context, log_file_path)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def save_best_checkpoint(
    output_dir: str, picsellia_model: Model, experiment: Experiment
):
    """
    Trouve le meilleur checkpoint, le copie dans exported_weights_dir, et le loggue dans l'expérience.
    """
    checkpoint_dirs = [
        d
        for d in glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if os.path.isdir(d)
    ]
    if not checkpoint_dirs:
        print("❌ No checkpoint directory found.")
        return

    # On prend le checkpoint avec le plus haut numéro
    best_ckpt = max(checkpoint_dirs, key=lambda p: int(p.split("-")[-1]))

    picsellia_model.exported_weights_path = best_ckpt

    # Logging sur l’expérience
    print("📦 Logging best checkpoint: model-latest")
    experiment.store(
        name="model-latest", path=picsellia_model.exported_weights_path, do_zip=True
    )

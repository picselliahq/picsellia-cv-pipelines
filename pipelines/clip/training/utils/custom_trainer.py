import glob
import json
import os
import re
import subprocess
import sys

import torch
from picsellia.types.enums import LogType
from PIL import Image
from transformers import (
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from picsellia_cv_engine.core import CocoDataset, DatasetCollection
from picsellia_cv_engine.core.contexts import (
    LocalTrainingContext,
    PicselliaTrainingContext,
)
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel
from utils.caption_generator import coco_to_rich_captions

class ClipModelTrainer:
    """
    CLIP model trainer using BLIP-generated captions for fine-tuning.
    """

    def __init__(
        self,
        model: CLIPModel,
        context: PicselliaTrainingContext | LocalTrainingContext,
    ):
        """
        Initialize the trainer.

        Args:
            model: The Picsellia model wrapper.
            context: Training context containing experiment, paths, and hyperparameters.
        """
        self.model = model
        self.context = context
        self.model_dir = os.path.join(model.results_dir, "clip_finetuned")
        os.makedirs(self.model_dir, exist_ok=True)
        self.run_script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "clip_utils.py"
        )

    def train_model(self, dataset_collection: DatasetCollection) -> CLIPModel:
        """
        Run the full CLIP fine-tuning process using BLIP captions.

        Args:
            dataset_collection: Collection with train, validation, and test datasets.

        Returns:
            The trained model with exported weights set.
        """
        working_dir = self.context.working_dir
        os.makedirs(json_dir := os.path.join(working_dir, "json"), exist_ok=True)

        json_files = {
            split: os.path.join(json_dir, f"{split}.json")
            for split in ["train", "val", "test"]
        }

        for split, json_path in json_files.items():
            export_dataset_to_clip_json(
                dataset=dataset_collection[split],
                output_path=json_path,
            )

        torch.cuda.empty_cache()

        os.makedirs(self.model_dir, exist_ok=True)

        run_clip_training(
            run_script_path=self.run_script_path,
            output_dir=self.model_dir,
            train_json=json_files["train"],
            val_json=json_files["val"],
            test_json=json_files["test"],
            batch_size=self.context.hyperparameters.batch_size,
            epochs=self.context.hyperparameters.epochs,
            context=self.context,
        )

        self.save_best_checkpoint(output_dir=self.model_dir, context=self.context)
        return self.model

    def save_best_checkpoint(
        self,
        output_dir: str,
        context: PicselliaTrainingContext | LocalTrainingContext,
    ) -> None:
        """
        Save the best checkpoint by selecting the latest one.

        Args:
            output_dir: Directory where checkpoints are stored.
            context: Training context for logging.
        """
        checkpoint_dirs = [
            d
            for d in glob.glob(os.path.join(output_dir, "checkpoint-*"))
            if os.path.isdir(d)
        ]
        if not checkpoint_dirs:
            print("❌ No checkpoint directory found.")
            return

        best_ckpt = max(checkpoint_dirs, key=lambda p: int(p.split("-")[-1]))
        print(f"📦 Saving best checkpoint: {os.path.basename(best_ckpt)}")
        context.experiment.store(name="model-latest", path=best_ckpt, do_zip=True)
        self.model.trained_weights_path = best_ckpt


def prepare_caption_model() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the BLIP processor and model for caption generation.

    Returns:
        A tuple containing the model and processor.
    """
    processor = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl"
    )
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl", device_map="auto"
    ).eval()
    return model, processor



def generate_caption(
    model: PreTrainedModel,
    processor: PreTrainedTokenizer,
    image_path: str,
    prompt: str,
    device: str,
) -> str:
    """
    Generate a caption from an image using BLIP.

    Args:
        model: Captioning model.
        processor: Processor for BLIP input formatting.
        image_path: Path to the image.
        prompt: Prompt to guide the captioning.
        device: Target device.

    Returns:
        A string caption.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {image_path}") from e

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    if not caption.endswith((".", "!", "?")):
        sentences = re.split(r"(?<=[.!?])\s+", caption)
        if len(sentences) > 1:
            caption = " ".join(sentences[:-1])

    return caption


def export_dataset_to_clip_json(
    dataset: CocoDataset,
    output_path: str,
) -> None:
    """
    Convert a COCO-format dataset to a JSONL file for CLIP training.

    Args:
        dataset: Dataset to process.
        output_path: Where to save the JSONL file.
    """
    print(f"📝 Generating captions for {dataset.name} dataset...")
    enriched_images = []

    enriched_images = coco_to_rich_captions(
        coco_json_path=dataset.coco_file_path,
        images_root=dataset.images_dir,
        max_captions_per_image=4,   
    )

    with open(output_path, "w") as f:
        for item in enriched_images:
            image_path = item["image"]
            captions = item["captions"]

            if isinstance(captions, str):
                captions = [captions]

            for caption in captions:
                json_line = {"image": image_path, "captions": caption}
                f.write(json.dumps(json_line, separators=(",", ":")) + "\n")



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
    """
    Build CLI command for CLIP training.

    Returns:
        List of command-line arguments.
    """
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
        "captions",
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


def parse_and_log_training_output(
    process: subprocess.Popen[str],
    context: PicselliaTrainingContext | LocalTrainingContext,
    log_file_path: str,
) -> None:
    """
    Parse stdout of subprocess and log relevant training metrics.

    Args:
        process: Running training process.
        context: Training context to log metrics.
        log_file_path: Path to write full logs.
    """
    train_pattern = re.compile(
        r"\{.*?'loss':\s*([\d.eE+-]+),\s*'grad_norm':\s*([\d.eE+-]+),"
        r"\s*'learning_rate':\s*([\d.eE+-]+),\s*'epoch':\s*([\d.]+).*?\}"
    )
    metrics_pattern = re.compile(r"'(\w+)'[\s]*:[\s]*([\d.eE+-]+)")

    with open(log_file_path, "w") as log_file:
        if process.stdout is None:
            raise RuntimeError("process.stdout is None. Cannot read training output.")

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

            match = train_pattern.search(line)
            if match:
                loss, grad_norm, lr, epoch = map(float, match.groups())
                context.experiment.log("train/loss", loss, LogType.LINE)
                context.experiment.log("train/grad_norm", grad_norm, LogType.LINE)
                context.experiment.log("train/learning_rate", lr, LogType.LINE)
            elif "'eval_loss'" in line and "'epoch'" in line:
                metrics = dict(metrics_pattern.findall(line))
                if "eval_loss" in metrics:
                    context.experiment.log(
                        "val/loss", float(metrics["eval_loss"]), LogType.LINE
                    )


def run_clip_training(
    run_script_path: str,
    output_dir: str,
    train_json: str,
    val_json: str,
    test_json: str,
    batch_size: int,
    epochs: int,
    context: PicselliaTrainingContext | LocalTrainingContext,
) -> None:
    """
    Run CLIP training with provided hyperparameters and log the output.

    Args:
        run_script_path: Path to training script.
        output_dir: Output directory for results.
        train_json: Path to training JSON file.
        val_json: Path to validation JSON file.
        test_json: Path to test JSON file.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        context: Context holding hyperparameters and experiment.
    """
    command = build_clip_command(
        model_name_or_path=context.hyperparameters.model_name,
        script_path=run_script_path,
        output_dir=output_dir,
        train_file=train_json,
        val_file=val_json,
        test_file=test_json,
        epochs=epochs,
        batch_size=batch_size,
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

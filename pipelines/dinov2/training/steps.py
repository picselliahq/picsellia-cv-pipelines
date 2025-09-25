import json
import os
import shutil

import torch
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection, Model
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)


class PicselliaImageDataset(Dataset):
    """
    Wraps a Picsellia CocoDataset into a PyTorch Dataset compatible with Hugging Face Trainer.

    - Utilise dataset_context.images_dir pour localiser les images.
    - Utilise dataset_context.coco_data pour récupérer les annotations et catégories.
    - Mappe category_id -> category_name -> label_id pour Hugging Face.
    """

    def __init__(self, dataset_context, processor, label2id):
        self.images_dir = dataset_context.images_dir
        self.processor = processor
        self.label2id = label2id

        coco = dataset_context.coco_data

        # index filename -> image_id
        self.image_id_by_filename = {
            img["file_name"]: img["id"] for img in coco["images"]
        }

        # index image_id -> category_id (1 annotation par image)
        self.category_id_by_image_id = {
            ann["image_id"]: ann["category_id"] for ann in coco["annotations"]
        }

        # index category_id -> category_name
        self.categories_by_id = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # liste des fichiers dispo
        self.filenames = list(self.image_id_by_filename.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.images_dir, filename)

        image = Image.open(filepath).convert("RGB")

        image_id = self.image_id_by_filename[filename]
        category_id = self.category_id_by_image_id[image_id]

        category_name = self.categories_by_id[category_id]
        label = self.label2id[category_name]

        encoding = self.processor(images=image, return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["labels"] = label

        return encoding


@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[CocoDataset]):
    context = Pipeline.get_active_context()

    # Load processor (normalization, resizing, etc.)
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)

    # Build label mapping from train set
    labels = list(picsellia_datasets["train"].labelmap.keys())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Prepare train / val datasets
    train_dataset = PicselliaImageDataset(
        picsellia_datasets["train"], processor, label2id
    )
    val_dataset = PicselliaImageDataset(picsellia_datasets["val"], processor, label2id)

    # Load pretrained DINOv2 for classification head
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    import evaluate

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=picsellia_model.results_dir,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=context.hyperparameters.learning_rate,
        per_device_train_batch_size=context.hyperparameters.batch_size,
        per_device_eval_batch_size=context.hyperparameters.batch_size,
        num_train_epochs=context.hyperparameters.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(picsellia_model.results_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save best model
    best_model_dir = os.path.join(picsellia_model.results_dir, "best_model")
    trainer.save_model(best_model_dir)

    # Créer un zip du dossier best_model
    best_model_zip = best_model_dir + ".zip"
    shutil.make_archive(best_model_dir, "zip", best_model_dir)

    # Uploader le zip comme artefact
    picsellia_model.save_artifact_to_experiment(
        experiment=context.experiment,
        artifact_name="best-model",
        artifact_path=best_model_zip,
    )


@step()
def evaluate_model(picsellia_model: Model, test_dataset: CocoDataset):
    best_model_dir = os.path.join(picsellia_model.results_dir, "best_model")
    processor = AutoImageProcessor.from_pretrained(best_model_dir)
    model = AutoModelForImageClassification.from_pretrained(best_model_dir)
    model.eval()

    results = []
    for filename in tqdm(
        os.listdir(test_dataset.images_dir), desc="Running inference on test set"
    ):
        filepath = os.path.join(test_dataset.images_dir, filename)

        image = Image.open(filepath).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = logits.argmax(-1).item()
            pred_label = model.config.id2label[pred_id]
            confidence = probs[0, pred_id].item()

        results.append(
            {"filename": filename, "pred_label": pred_label, "confidence": confidence}
        )

    results_file = os.path.join(
        picsellia_model.results_dir, "test_inference_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results_file

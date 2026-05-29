import os

import torch
from picsellia.types.enums import InferenceType
from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection
from picsellia_cv_engine.core.contexts import PicselliaTrainingContext
from picsellia_cv_engine.core.services.model.utils import evaluate_model_impl

from utils.training import (
    load_sam3_model_and_processor,
    run_sam3_inference,
    train_sam3,
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@step
def load_sam3_model():
    """Load the pretrained SAM-3 model and processor from Hugging Face."""
    return load_sam3_model_and_processor(device=_device())


@step
def train_sam3_model(
    model, processor, datasets: DatasetCollection[CocoDataset]
) -> None:
    """Fine-tune SAM-3 on the training split and store the weights on the experiment."""
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    output_dir = context.working_dir
    os.makedirs(output_dir, exist_ok=True)

    train_sam3(
        model=model,
        processor=processor,
        datasets=datasets,
        hp=context.hyperparameters,
        experiment=context.experiment,
        output_dir=output_dir,
        device=_device(),
    )


@step
def evaluate_sam3_model(
    model, processor, datasets: DatasetCollection[CocoDataset]
) -> None:
    """Run the fine-tuned model on the test split and evaluate it in Picsellia."""
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    dataset = datasets["test"]

    predictions = run_sam3_inference(
        model=model,
        processor=processor,
        dataset=dataset,
        hp=context.hyperparameters,
        device=_device(),
    )

    if not predictions:
        print("No predictions produced on the test split; skipping evaluation.")
        return

    try:
        training_labelmap = context.experiment.get_log("labelmap").data
    except Exception:
        training_labelmap = {
            str(i): name for i, name in enumerate(dataset.labelmap.keys())
        }

    evaluate_model_impl(
        context=context,
        picsellia_predictions=predictions,
        inference_type=InferenceType.SEGMENTATION,
        assets=dataset.assets,
        output_dir=os.path.join(context.working_dir, "evaluation"),
        training_labelmap=training_labelmap,
    )

from picsellia_cv_engine import Pipeline, step
from picsellia_cv_engine.core import CocoDataset, DatasetCollection
from picsellia_cv_engine.frameworks.clip.model.model import CLIPModel
from utils.custom_trainer import ClipModelTrainer

@step()
def train(
    model: CLIPModel, dataset_collection: DatasetCollection[CocoDataset]
) -> CLIPModel:
    """
    Training step for CLIP using the Picsellia training engine.

    This step uses BLIP to generate captions for the dataset and runs CLIP fine-tuning using a CLI script.

    Args:
        model: CLIP model instance to be trained.
        dataset_collection: Dataset collection containing train/val/test splits.

    Returns:
        The trained CLIP model with updated weights.
    """
    context = Pipeline.get_active_context()
    trainer = ClipModelTrainer(
        model=model,
        context=context,
    )
    model = trainer.train_model(dataset_collection=dataset_collection)
    return model

from picsellia_cv_engine.core.services.utils.picsellia_context import (
    create_picsellia_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_dataset_annotations
from picsellia_cv_engine.steps.base.model.builder import build_model

from pipelines.grounding_dino.steps import process
from pipelines.grounding_dino.utils.parameters import GroundingDinoProcessingParameters

processing_context = create_picsellia_processing_context(
    processing_parameters_cls=GroundingDinoProcessingParameters,
)


@pipeline(
    context=processing_context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def grounding_dino_pipeline():
    picsellia_dataset = load_coco_datasets()
    picsellia_model = build_model(
        pretrained_weights_name="pretrained-weights", config_name="config"
    )
    picsellia_dataset = process(
        picsellia_model=picsellia_model, picsellia_dataset=picsellia_dataset
    )
    upload_dataset_annotations(dataset=picsellia_dataset, use_id=True)


if __name__ == "__main__":
    grounding_dino_pipeline()

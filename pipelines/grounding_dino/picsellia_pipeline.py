from picsellia_cv_engine.core.services.utils.picsellia_context import (
    create_picsellia_processing_context,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_dataset_annotations
from picsellia_cv_engine.steps.base.model.prediction_converter import (
    convert_predictions_to_coco,
)
from picsellia_cv_engine.steps.grounding_dino.model.loader import (
    load_grounding_dino_model,
)
from picsellia_cv_engine.steps.grounding_dino.model.predictor import (
    run_grounding_dino_inference,
)
from utils.parameters import ProcessingParameters

processing_context = create_picsellia_processing_context(
    processing_parameters_cls=ProcessingParameters,
)


@pipeline(
    context=processing_context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def grounding_dino_pipeline():
    picsellia_dataset = load_coco_datasets()
    grounding_dino_model = load_grounding_dino_model(
        pretrained_weights_name="pretrained-weights", config_name="config"
    )
    predictions = run_grounding_dino_inference(
        model=grounding_dino_model,
        dataset=picsellia_dataset,
    )
    picsellia_dataset = convert_predictions_to_coco(
        predictions=predictions,
        dataset=picsellia_dataset,
    )
    upload_dataset_annotations(dataset=picsellia_dataset, use_id=True)


if __name__ == "__main__":
    grounding_dino_pipeline()

import argparse
import sys

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
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

if not hasattr(sys.stderr, "isatty") or not hasattr(sys.stdout, "isatty"):
    try:
        sys.stderr.isatty = lambda: False
        sys.stdout.isatty = lambda: False
    except Exception:
        pass

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.PRE_ANNOTATION,
    processing_parameters_cls=ProcessingParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
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
        predictions=predictions, dataset=picsellia_dataset, use_id=True
    )
    upload_dataset_annotations(dataset=picsellia_dataset, use_id=True)


if __name__ == "__main__":
    grounding_dino_pipeline()

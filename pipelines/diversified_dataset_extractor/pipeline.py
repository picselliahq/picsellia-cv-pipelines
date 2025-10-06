import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from steps import load_model, process, validate_data, validate_weights
from utils.parameters import ProcessingDiversifiedDataExtractorParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.DATASET_VERSION_CREATION,
    processing_parameters_cls=ProcessingDiversifiedDataExtractorParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def diversified_data_extractor_pipeline() -> None:
    datasets = load_coco_datasets(skip_asset_listing=True)

    validate_data(dataset=datasets["input"])
    pretrained_weights = validate_weights()
    embedding_model = load_model(pretrained_weights=pretrained_weights)

    process(
        input_dataset=datasets["input"],
        output_dataset=datasets["output"],
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    diversified_data_extractor_pipeline()

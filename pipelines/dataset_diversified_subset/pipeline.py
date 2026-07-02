import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import (
    create_processing_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from steps import (
    create_subset_dataset_version,
    fetch_dataset_embeddings,
    select_diverse_subset,
)
from utils.parameters import ProcessingParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_processing_context_from_config(
    processing_type=ProcessingType.DATASET_VERSION_CREATION,
    processing_parameters_cls=ProcessingParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)

@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def dataset_diversified_subset_pipeline():
    data_ids, vectors = fetch_dataset_embeddings()
    selected_data_ids = select_diverse_subset(data_ids=data_ids, vectors=vectors)
    create_subset_dataset_version(selected_data_ids=selected_data_ids)


if __name__ == "__main__":
    dataset_diversified_subset_pipeline()

import argparse

from picsellia.types.enums import ProcessingType
from picsellia_cv_engine.core.services.context.unified_context import create_processing_context_from_config
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline


from steps import length_dataset_version_sanity_check, parameters_sanity_chek, split_and_tag_data, create_empty_annotation
from utils.parameters import ProcessingParameters

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
def split_train_test_val_pipeline():
    length_dataset_version_sanity_check()
    parameters_sanity_chek()
    create_empty_annotation()
    split_and_tag_data()

if __name__ == "__main__":
    split_train_test_val_pipeline()

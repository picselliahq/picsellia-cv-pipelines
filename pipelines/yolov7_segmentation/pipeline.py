import argparse

from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.core.services.context.unified_context import (
    create_training_context_from_config,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.validator import validate_dataset
from steps import (
    get_dataset_collection,
    get_model,
    prepare_dataset_collection,
    prepare_model,
    train_model,
)
from utils.parameters import Yolov7AugmentationParameters, Yolov7HyperParameters

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["local", "picsellia"], default="picsellia")
parser.add_argument("--config-file", type=str, required=False)
args = parser.parse_args()

context = create_training_context_from_config(
    hyperparameters_cls=Yolov7HyperParameters,
    augmentation_parameters_cls=Yolov7AugmentationParameters,
    export_parameters_cls=ExportParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)


@pipeline(
    context=context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov7_segmentation_training_pipeline():
    dataset_collection = get_dataset_collection()
    dataset_collection = prepare_dataset_collection(
        dataset_collection=dataset_collection
    )
    validate_dataset(dataset=dataset_collection, fix_annotation=True)

    model = get_model(
        pretrained_weights_name="pretrained-weights",
        config_name="config",
        hyperparameters_name="hyperparameters",
    )
    # model = load_model()
    model = prepare_model(model=model)
    model = train_model(model=model, dataset_collection=dataset_collection)
    # evaluate_yolov7_model(model=model, dataset=dataset_collection["test"])


if __name__ == "__main__":
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()

    yolov7_segmentation_training_pipeline()

# type: ignore

from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.dataset.loader import load_coco_datasets
from picsellia_cv_engine.steps.base.dataset.uploader import upload_dataset_annotations

from pipelines.yolov8.pre_annotation.pipeline_utils.parameters.processing_yolov8_preannotation_parameters import (
    ProcessingYOLOV8PreannotationParameters,
)
from pipelines.yolov8.pre_annotation.pipeline_utils.steps.model_loading.processing_ultralytics_model_loader import (
    load_processing_ultralytics_model,
)
from pipelines.yolov8.pre_annotation.pipeline_utils.steps.processing.yolov8_preannotation_processing import (
    process,
)
from pipelines.yolov8.pre_annotation.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_processing_ultralytics_model,
)


def get_context() -> PicselliaProcessingContext[
    ProcessingYOLOV8PreannotationParameters
]:
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingYOLOV8PreannotationParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def yolov8_preannotation_processing_pipeline() -> None:
    dataset = load_coco_datasets()
    model = get_processing_ultralytics_model()
    load_processing_ultralytics_model(
        model=model,
        weights_path_to_load=model.trained_weights_path,
    )
    output_dataset = process(model=model, dataset=dataset)
    upload_dataset_annotations(dataset=output_dataset)


if __name__ == "__main__":
    yolov8_preannotation_processing_pipeline()

# type: ignore

from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.steps.data_extraction.processing_data_extractor import (
    get_processing_dataset_context,
)
from picsellia_cv_engine.steps.data_upload.annotations_uploader import (
    upload_annotations,
)

from pipelines.yolov8_preannotation.pipeline_utils.parameters.processing_yolov8_preannotation_parameters import (
    ProcessingYOLOV8PreannotationParameters,
)
from pipelines.yolov8_preannotation.pipeline_utils.steps.model_loading.processing_ultralytics_model_loader import (
    load_processing_ultralytics_model_context,
)
from pipelines.yolov8_preannotation.pipeline_utils.steps.processing.yolov8_preannotation_processing import (
    process,
)
from pipelines.yolov8_preannotation.pipeline_utils.steps.weights_extraction.ultralytics_weights_extractor import (
    get_processing_ultralytics_model_context,
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
    dataset_context = get_processing_dataset_context()
    model_context = get_processing_ultralytics_model_context()
    load_processing_ultralytics_model_context(
        model_context=model_context,
        weights_path_to_load=model_context.trained_weights_path,
    )
    output_dataset_context = process(
        model_context=model_context, dataset_context=dataset_context
    )
    upload_annotations(dataset_context=output_dataset_context)


if __name__ == "__main__":
    yolov8_preannotation_processing_pipeline()

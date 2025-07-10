from picsellia_cv_engine.core.contexts.processing.datalake.picsellia_datalake_processing_context import (
    PicselliaDatalakeProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.base.datalake.loader import load_datalake

from pipelines.datalake_autotagging.pipeline_utils.parameters.processing_datalake_autotagging_parameters import (
    ProcessingDatalakeAutotaggingParameters,
)
from pipelines.datalake_autotagging.pipeline_utils.steps.model_loading.clip_model_context_loader import (
    load_clip_model,
)
from pipelines.datalake_autotagging.pipeline_utils.steps.processing.clip_datalake_autotagging import (
    autotag_datalake_with_clip,
)
from pipelines.datalake_autotagging.pipeline_utils.steps.weights_extraction.hugging_face_weights_extractor import (
    get_hugging_face_model,
)


def get_context() -> PicselliaDatalakeProcessingContext[
    ProcessingDatalakeAutotaggingParameters
]:
    return PicselliaDatalakeProcessingContext(
        processing_parameters_cls=ProcessingDatalakeAutotaggingParameters
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def datalake_autotagging_processing_pipeline() -> None:
    datalake = load_datalake()
    model = get_hugging_face_model()
    model = load_clip_model(model=model, device="cuda:0")
    autotag_datalake_with_clip(datalake=datalake, model=model, device="cuda:0")


if __name__ == "__main__":
    import os

    import torch

    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count > 1:
        torch.set_num_threads(cpu_count - 1)

    datalake_autotagging_processing_pipeline()

from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.datalake_autotagging.pipeline_utils.model.hugging_face_model import (
    HuggingFaceModel,
)
from pipelines.datalake_autotagging.pipeline_utils.steps_utils.model_loading.clip_model_loader import (
    clip_load_model,
)


@step
def load_clip_model(
    model: HuggingFaceModel,
    device: str = "cuda:0",
) -> HuggingFaceModel:
    loaded_model, loaded_processor = clip_load_model(
        model_name=model.hugging_face_model_name,
        device=device,
    )
    model.set_loaded_model(loaded_model)
    model.set_loaded_processor(loaded_processor)
    return model

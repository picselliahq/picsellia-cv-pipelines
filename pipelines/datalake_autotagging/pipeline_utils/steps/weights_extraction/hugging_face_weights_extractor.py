import logging

from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.datalake_autotagging.pipeline_utils.model.hugging_face_model import (
    HuggingFaceModel,
)

logger = logging.getLogger(__name__)


@step
def get_hugging_face_model(
    hugging_face_model_name: str | None = None,
) -> HuggingFaceModel:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    model_version = context.model_version
    if not hugging_face_model_name:
        model_parameters = model_version.sync()["parameters"]
        hugging_face_model_name = model_parameters.get("hugging_face_model_name")
        if not hugging_face_model_name:
            raise ValueError(
                "Hugging Face model name not provided. Please provide it as an argument or set the 'hugging_face_model_name' parameter in the model version."
            )
    print(f"Loading Hugging Face model {hugging_face_model_name}")
    model = HuggingFaceModel(
        hugging_face_model_name=hugging_face_model_name,
        model_name=model_version.name,
        model_version=model_version,
        pretrained_weights_name=None,
        trained_weights_name=None,
        config_name=None,
        exported_weights_name=None,
    )
    return model

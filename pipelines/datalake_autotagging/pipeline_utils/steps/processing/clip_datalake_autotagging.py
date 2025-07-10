import logging

from picsellia_cv_engine.core import (
    Datalake,
    DatalakeCollection,
)
from picsellia_cv_engine.core.contexts.processing.dataset.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from pipelines.datalake_autotagging.pipeline_utils.model.hugging_face_model import (
    HuggingFaceModel,
)
from pipelines.datalake_autotagging.pipeline_utils.steps_utils.model_prediction.clip_model_predictor import (
    CLIPModelPredictor,
)


@step
def autotag_datalake_with_clip(
    datalake: Datalake | DatalakeCollection,
    model: HuggingFaceModel,
    device: str = "cuda:0",
):
    context: PicselliaProcessingContext = Pipeline.get_active_context()

    model_predictor = CLIPModelPredictor(
        model=model,
        tags_list=context.processing_parameters.tags_list,
        device=device,
    )
    if isinstance(datalake, Datalake):
        datalake = datalake
    elif isinstance(datalake, DatalakeCollection):
        datalake = datalake["input"]
    else:
        raise ValueError("Datalake should be either a Datalake or a DatalakeCollection")

    image_inputs, image_paths = model_predictor.pre_process_datalake(
        datalake=datalake,
    )
    image_input_batches = model_predictor.prepare_batches(
        images=image_inputs,
        batch_size=context.processing_parameters.batch_size,
    )
    image_path_batches = model_predictor.prepare_batches(
        images=image_paths,
        batch_size=context.processing_parameters.batch_size,
    )
    batch_results = model_predictor.run_inference_on_batches(
        image_batches=image_input_batches
    )
    picsellia_datalake_autotagging_predictions = model_predictor.post_process_batches(
        image_batches=image_path_batches,
        batch_results=batch_results,
        datalake=datalake,
    )
    logging.info(f"Predictions for datalake {datalake.datalake.id} done.")

    for (
        picsellia_datalake_autotagging_prediction
    ) in picsellia_datalake_autotagging_predictions:
        if not picsellia_datalake_autotagging_prediction["tag"]:
            continue
        picsellia_datalake_autotagging_prediction["data"].add_tags(
            tags=picsellia_datalake_autotagging_prediction["tag"]
        )

    logging.info(f"Tags added to datalake {datalake.datalake.id}.")

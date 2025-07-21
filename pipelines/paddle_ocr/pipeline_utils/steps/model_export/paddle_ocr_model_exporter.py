# type: ignore
from picsellia_cv_engine.core.contexts import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step

from paddle_ocr.pipeline_utils.model.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from paddle_ocr.pipeline_utils.parameters.paddle_ocr_augmentation_parameters import (
    PaddleOCRAugmentationParameters,
)
from paddle_ocr.pipeline_utils.parameters.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)
from paddle_ocr.pipeline_utils.steps_utils.model_export.paddle_ocr_model_collection_exporter import (
    PaddleOCRModelCollectionExporter,
)


@step
def export_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
) -> PaddleOCRModelCollection:
    """
    Exports a PaddleOCR model collection and saves it to an experiment.

    This function retrieves the active training context from the pipeline, exports the provided
    PaddleOCR model collection in the specified format, and saves the exported models to the experiment.
    The `PaddleOCRModelCollectionExporter` is used to handle the export and save operations.

    Args:
        model_collection (PaddleOCRModelCollection): The PaddleOCR model collection to be exported.

    Returns:
        PaddleOCRModelCollection: The exported PaddleOCR model collection.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()
    model_collection_exporter = PaddleOCRModelCollectionExporter(
        model_collection=model_collection
    )
    model_collection = model_collection_exporter.export_model_collection(
        export_format=context.export_parameters.export_format
    )
    model_collection_exporter.save_model_collection(experiment=context.experiment)

    return model_collection

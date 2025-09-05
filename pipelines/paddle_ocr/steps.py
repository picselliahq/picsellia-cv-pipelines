import os

from picsellia_cv_engine.core import (
    CocoDataset,
    DatasetCollection,
)
from picsellia_cv_engine.core.contexts import (
    PicselliaTrainingContext,
)
from picsellia_cv_engine.core.data import (
    TBaseDataset,
)
from picsellia_cv_engine.core.models import Model
from picsellia_cv_engine.core.parameters.export_parameters import (
    ExportParameters,
)
from picsellia_cv_engine.core.services.model.evaluator.model_evaluator import (
    ModelEvaluator,
)
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline
from picsellia_cv_engine.decorators.step_decorator import step
from utils.dataset import PaddleOCRDataset
from utils.dataset_preparation import PaddleOCRDatasetPreparator
from utils.model import PaddleOCRModelCollection
from utils.model_export import PaddleOCRModelCollectionExporter
from utils.model_loading import paddle_ocr_load_model
from utils.model_prediction import PaddleOCRModelCollectionPredictor
from utils.model_preparation import PaddleOCRModelCollectionPreparator
from utils.model_training import PaddleOCRModelCollectionTrainer
from utils.parameters import PaddleOCRAugmentationParameters, PaddleOCRHyperParameters


@step
def prepare_paddle_ocr_dataset_collection(
    dataset_collection: DatasetCollection[CocoDataset],
) -> DatasetCollection[PaddleOCRDataset]:
    """
    Prepares and organizes a dataset collection for PaddleOCR training.

    This function takes an existing `DatasetCollection` containing the 'train', 'val', and 'test' datasets,
    and organizes them into a format suitable for PaddleOCR training. It uses the `PaddleOCRDatasetPreparator`
    to organize the datasets (e.g., creating necessary directories and moving images) for each dataset split (train, val, test).
    The organized datasets are then stored in a new `DatasetCollection` with `PaddleOCRDataset` types.

    Args:
        dataset_collection (DatasetCollection[CocoDataset]): The original dataset collection containing 'train', 'val', and 'test' splits.

    Returns:
        DatasetCollection[PaddleOCRDataset]: A new dataset collection where each dataset is organized for PaddleOCR,
        with directories properly set up for training, validation, and testing.
    """
    context = Pipeline.get_active_context()

    paddleocr_dataset_collection = DatasetCollection(
        [
            PaddleOCRDatasetPreparator(
                dataset=dataset_collection["train"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["train"].name,
                    )
                ),
            ).organize(),
            PaddleOCRDatasetPreparator(
                dataset=dataset_collection["val"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["val"].name,
                    )
                ),
            ).organize(),
            PaddleOCRDatasetPreparator(
                dataset=dataset_collection["test"],
                destination_path=str(
                    os.path.join(
                        os.getcwd(),
                        context.experiment.name,
                        "dataset",
                        dataset_collection["test"].name,
                    )
                ),
            ).organize(),
        ]
    )

    return paddleocr_dataset_collection


@step
def get_paddle_ocr_model_collection() -> PaddleOCRModelCollection:
    """
    Extracts a PaddleOCR model collection from a Picsellia experiment.

    This function retrieves the active training context and extracts the base model version from the experiment.
    It creates two `Model` objects for the bounding box detection model ("bbox-model") and the text
    recognition model ("text-model"), specifying their configurations and pretrained weights. The function
    then downloads the necessary model weights and returns the `PaddleOCRModelCollection` containing both models.

    Returns:
        PaddleOCRModelCollection: The extracted and initialized PaddleOCR model collection with both the
        bounding box and text recognition models.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_version = context.experiment.get_base_model_version()

    bbox_model = Model(
        name="bbox-model",
        model_version=model_version,
        pretrained_weights_name="bbox-pretrained-model",
        trained_weights_name=None,
        config_name="bbox-config",
        exported_weights_name=None,
    )
    text_model = Model(
        name="text-model",
        model_version=model_version,
        pretrained_weights_name="text-pretrained-model",
        trained_weights_name=None,
        config_name="text-config",
        exported_weights_name=None,
    )

    model_collection = PaddleOCRModelCollection(
        bbox_model=bbox_model, text_model=text_model
    )
    model_collection.download_weights(
        destination_dir=os.path.join(os.getcwd(), context.experiment.name, "model")
    )

    return model_collection


@step
def prepare_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
    dataset_collection: DatasetCollection[PaddleOCRDataset],
) -> PaddleOCRModelCollection:
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()
    model_collection_preparator = PaddleOCRModelCollectionPreparator(
        model_collection=model_collection,
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
    )
    model_collection = model_collection_preparator.prepare()
    return model_collection


@step
def load_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
) -> PaddleOCRModelCollection:
    """
    Loads a PaddleOCR model collection from pretrained weights if available.

    This function retrieves the active training context and attempts to load the PaddleOCR model collection
    (both bounding box and text recognition models) from their respective pretrained weights directories.
    The function checks for the existence of the required weight files and the character dictionary. If all files
    are present, the models are loaded onto the specified device. If any required files are missing,
    a `FileNotFoundError` is raised.

    Args:
        model_collection (PaddleOCRModelCollection): The PaddleOCR model collection to load pretrained weights into.

    Returns:
        PaddleOCRModelCollection: The model collection with the loaded models.

    Raises:
        FileNotFoundError: If any of the required model weight files or the character dictionary file are not found.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    if (
        model_collection.bbox_model.exported_weights_dir
        and model_collection.text_model.exported_weights_dir
        and os.path.exists(model_collection.bbox_model.exported_weights_dir)
        and os.path.exists(model_collection.text_model.exported_weights_dir)
        and os.path.exists(
            os.path.join(model_collection.text_model.weights_dir, "en_dict.txt")
        )
    ):
        loaded_model = paddle_ocr_load_model(
            bbox_model_path_to_load=model_collection.bbox_model.exported_weights_dir,
            text_model_path_to_load=model_collection.text_model.exported_weights_dir,
            character_dict_path_to_load=os.path.join(
                model_collection.text_model.weights_dir, "en_dict.txt"
            ),
            device=context.hyperparameters.device,
        )
        model_collection.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {model_collection.bbox_model.exported_weights_dir} or {model_collection.text_model.exported_weights_dir}. Cannot load model."
        )

    return model_collection


@step
def train_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
) -> PaddleOCRModelCollection:
    """
    Trains a PaddleOCR model collection based on the provided hyperparameters.

    This function retrieves the active training context from the pipeline and initializes a
    `PaddleOCRModelCollectionTrainer`. It then trains the model collection, including both the
    bounding box detection and text recognition models, for the number of epochs specified in the
    hyperparameters. After training, the updated model collection is returned.

    Args:
        model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models (bounding box and text recognition) to be trained.

    Returns:
        PaddleOCRModelCollection: The trained model collection.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_trainer = PaddleOCRModelCollectionTrainer(
        model_collection=model_collection, experiment=context.experiment
    )

    model_collection = model_trainer.train_model_collection(
        bbox_epochs=context.hyperparameters.bbox_epochs,
        text_epochs=context.hyperparameters.text_epochs,
    )

    return model_collection


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


@step
def evaluate_paddle_ocr_model_collection(
    model_collection: PaddleOCRModelCollection,
    dataset: TBaseDataset,
) -> None:
    """
    Evaluates a PaddleOCR model collection on a given dataset.

    This function retrieves the active training context from the pipeline, performs inference using
    the provided PaddleOCR model collection on the dataset, and evaluates the predictions. It processes
    the dataset in batches, runs inference, and then logs the evaluation results to the experiment.

    Args:
        model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models to be evaluated.
        dataset (TDataset): The dataset containing the data for evaluation.

    Returns:
        None: The function performs evaluation and logs the results but does not return any value.
    """
    context: PicselliaTrainingContext[
        PaddleOCRHyperParameters, PaddleOCRAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_collection_predictor = PaddleOCRModelCollectionPredictor(
        model_collection=model_collection,
    )
    image_paths = model_collection_predictor.pre_process_dataset(dataset=dataset)
    image_batches = model_collection_predictor.prepare_batches(
        image_paths=image_paths,
        batch_size=min(
            context.hyperparameters.bbox_batch_size,
            context.hyperparameters.text_batch_size,
        ),
    )
    batch_results = model_collection_predictor.run_inference_on_batches(
        image_batches=image_batches
    )
    picsellia_ocr_predictions = model_collection_predictor.post_process_batches(
        image_batches=image_batches,
        batch_results=batch_results,
        dataset=dataset,
    )

    model_evaluator = ModelEvaluator(
        experiment=context.experiment,
        inference_type=model_collection.bbox_model.model_version.type,
    )
    model_evaluator.evaluate(picsellia_predictions=picsellia_ocr_predictions)

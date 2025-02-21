import os
import yaml

from src.picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from pipelines.paddle_ocr.pipeline_utils.dataset.paddle_ocr_dataset_context import (
    PaddleOCRDatasetContext,
)
from src.picsellia_cv_engine.models.model.model_context import ModelContext
from pipelines.paddle_ocr.pipeline_utils.model.paddle_ocr_model_collection import (
    PaddleOCRModelCollection,
)
from pipelines.paddle_ocr.pipeline_utils.parameters.paddle_ocr_hyper_parameters import (
    PaddleOCRHyperParameters,
)


def generate_bbox_yaml_config(
    dataset_collection: DatasetCollection[PaddleOCRDatasetContext],
    model_context: ModelContext,
    hyperparameters: PaddleOCRHyperParameters,
):
    """
    Generates a YAML configuration for the bounding box detection model in PaddleOCR.

    This function updates and modifies the YAML configuration for the bounding box model
    based on the dataset, model context, and provided hyperparameters.

    Args:
        dataset_collection (DatasetCollection): The collection of datasets for training and validation.
        model_context (ModelContext): The context of the model, including paths to pretrained and saved weights.
        hyperparameters (PaddleOCRHyperParameters): Hyperparameters to customize the training process.

    Returns:
        dict: The modified configuration dictionary.
    """
    if model_context.config_path is None:
        raise ValueError("No config file path provided for the model context")

    with open(model_context.config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["Global"]["use_gpu"] = (
        True if hyperparameters.device.startswith("cuda") else False
    )
    config["Global"]["epoch_num"] = hyperparameters.bbox_epochs
    config["Global"]["pretrained_model"] = model_context.pretrained_weights_path
    config["Global"]["save_model_dir"] = model_context.trained_weights_dir
    config["Global"]["save_res_path"] = model_context.results_dir
    config["Global"]["save_inference_dir"] = model_context.exported_weights_dir
    config["Global"]["save_epoch_step"] = hyperparameters.bbox_save_epoch_step

    config["Optimizer"]["lr"]["learning_rate"] = hyperparameters.bbox_learning_rate

    config["Train"]["dataset"]["name"] = "SimpleDataSet"
    config["Train"]["dataset"]["data_dir"] = dataset_collection["train"].images_dir
    config["Train"]["dataset"]["label_file_list"] = [
        dataset_collection["train"].paddle_ocr_bbox_annotations_path
    ]
    config["Train"]["loader"]["batch_size_per_card"] = hyperparameters.bbox_batch_size
    config["Train"]["loader"]["shuffle"] = True

    config["Eval"]["dataset"]["name"] = "SimpleDataSet"
    config["Eval"]["dataset"]["data_dir"] = dataset_collection["val"].images_dir
    config["Eval"]["dataset"]["label_file_list"] = [
        dataset_collection["val"].paddle_ocr_bbox_annotations_path
    ]
    config["Eval"]["loader"]["batch_size_per_card"] = hyperparameters.bbox_batch_size
    config["Eval"]["loader"]["shuffle"] = True

    return config


def generate_text_yaml_config(
    dataset_collection: DatasetCollection[PaddleOCRDatasetContext],
    model_context: ModelContext,
    hyperparameters: PaddleOCRHyperParameters,
):
    """
    Generates a YAML configuration for the text recognition model in PaddleOCR.

    This function updates and modifies the YAML configuration for the text model
    based on the dataset, model context, and provided hyperparameters.

    Args:
        dataset_collection (DatasetCollection): The collection of datasets for training and validation.
        model_context (ModelContext): The context of the model, including paths to pretrained and saved weights.
        hyperparameters (PaddleOCRHyperParameters): Hyperparameters to customize the training process.

    Returns:
        dict: The modified configuration dictionary.
    """
    if not model_context.config_path:
        raise ValueError("No config file path provided for the model context")

    if not model_context.weights_dir:
        raise ValueError("No weights directory provided for the model context")

    with open(model_context.config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config["Global"]["use_gpu"] = (
        True if hyperparameters.device.startswith("cuda") else False
    )
    config["Global"]["epoch_num"] = hyperparameters.text_epochs
    if model_context.pretrained_weights_path is not None:
        config["Global"]["pretrained_model"] = os.path.join(
            model_context.pretrained_weights_path, "best_accuracy"
        )
    config["Global"]["save_model_dir"] = model_context.trained_weights_dir
    config["Global"]["save_res_path"] = model_context.results_dir
    config["Global"]["save_inference_dir"] = model_context.exported_weights_dir
    config["Global"]["save_epoch_step"] = hyperparameters.text_save_epoch_step
    config["Global"]["max_text_length"] = hyperparameters.max_text_length
    config["Global"]["character_dict_path"] = os.path.join(
        model_context.weights_dir, "en_dict.txt"
    )

    if "SARHead" in config["Architecture"]["Head"]["head_list"]:
        config["Architecture"]["Head"]["head_list"]["SARHead"]["max_text_length"] = (
            hyperparameters.max_text_length
        )

    config["Optimizer"]["lr"]["learning_rate"] = hyperparameters.text_learning_rate

    config["Train"]["dataset"]["name"] = "SimpleDataSet"
    config["Train"]["dataset"]["data_dir"] = dataset_collection["train"].text_images_dir
    config["Train"]["dataset"]["label_file_list"] = [
        dataset_collection["train"].paddle_ocr_text_annotations_path
    ]
    config["Train"]["loader"]["batch_size_per_card"] = hyperparameters.text_batch_size
    config["Train"]["loader"]["shuffle"] = True

    if "RecConAug" in config["Train"]["dataset"]["transforms"]:
        config["Train"]["dataset"]["transforms"]["RecConAug"]["max_text_length"] = (
            hyperparameters.max_text_length
        )

    config["Eval"]["dataset"]["name"] = "SimpleDataSet"
    config["Eval"]["dataset"]["data_dir"] = dataset_collection["val"].text_images_dir
    config["Eval"]["dataset"]["label_file_list"] = [
        dataset_collection["val"].paddle_ocr_text_annotations_path
    ]
    config["Eval"]["loader"]["batch_size_per_card"] = hyperparameters.text_batch_size
    config["Eval"]["loader"]["shuffle"] = True

    return config


def write_yaml_config(config, config_file_path):
    """
    Writes the given YAML configuration to a file.

    Args:
        config (dict): The configuration dictionary to write.
        config_file_path (str): The file path where the configuration will be saved.
    """
    with open(config_file_path, "w") as file:
        yaml.dump(config, file)


class PaddleOCRModelCollectionPreparator:
    """
    Prepares the PaddleOCR model collection by generating and writing configuration files for each model.

    This class generates YAML configuration files for both the bounding box and text recognition models,
    and saves them to the appropriate paths.

    Attributes:
        model_collection (PaddleOCRModelCollection): The collection of PaddleOCR models.
        dataset_collection (DatasetCollection): The dataset collection containing training and validation datasets.
        hyperparameters (PaddleOCRHyperParameters): Hyperparameters for training the models.
    """

    def __init__(
        self,
        model_collection: PaddleOCRModelCollection,
        dataset_collection: DatasetCollection,
        hyperparameters: PaddleOCRHyperParameters,
    ):
        """
        Initializes the preparator with the model collection, dataset collection, and hyperparameters.

        Args:
            model_collection (PaddleOCRModelCollection): The collection of models to prepare.
            dataset_collection (DatasetCollection): The datasets for training and evaluation.
            hyperparameters (PaddleOCRHyperParameters): Hyperparameters to customize the model training process.
        """
        self.model_collection = model_collection
        self.dataset_collection = dataset_collection
        self.hyperparameters = hyperparameters

    def prepare(self):
        """
        Prepares the model collection by generating and writing YAML configurations for both models.

        Returns:
            PaddleOCRModelCollection: The prepared model collection.
        """
        bbox_config = generate_bbox_yaml_config(
            self.dataset_collection,
            self.model_collection.bbox_model,
            self.hyperparameters,
        )
        text_config = generate_text_yaml_config(
            self.dataset_collection,
            self.model_collection.text_model,
            self.hyperparameters,
        )
        write_yaml_config(bbox_config, self.model_collection.bbox_model.config_path)
        write_yaml_config(text_config, self.model_collection.text_model.config_path)
        return self.model_collection

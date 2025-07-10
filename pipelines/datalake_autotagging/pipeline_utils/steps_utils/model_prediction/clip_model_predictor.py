import os
from typing import Any
from uuid import UUID

from picsellia import Tag
from picsellia_cv_engine.core import Datalake
from picsellia_cv_engine.core.services.model.predictor.model_predictor import (
    ModelPredictor,
)
from PIL import Image

from pipelines.datalake_autotagging.pipeline_utils.model.hugging_face_model import (
    HuggingFaceModel,
)
from pipelines.datalake_autotagging.pipeline_utils.steps_utils.model_loading.clip_model_loader import (
    get_device,
)


def create_tags(datalake: Datalake, list_tags: list):
    """
    Creates or retrieves tags from the Datalake.
    Args:
        datalake (Datalake): The datalake object to interact with.
        list_tags (list): List of tags to create or retrieve.
    Returns:
        dict: A dictionary of tag names and Tag objects.
    """
    if list_tags:
        for tag_name in list_tags:
            datalake.get_or_create_data_tag(name=tag_name)
    return {k.name: k for k in datalake.list_data_tags()}


class CLIPModelPredictor(ModelPredictor[HuggingFaceModel]):
    """
    A class to handle the prediction process for CLIP model within a given model.
    Args:
        model (HuggingFaceModel): The model containing the HuggingFace model and processor.
        tags_list (List[str]): A list of tags used for image classification.
        device (str): The device ('cpu' or 'gpu') on which to run the model.
    """

    def __init__(
        self,
        model: HuggingFaceModel,
        tags_list: list[str],
        device: str = "cuda:0",
    ):
        """
        Initializes the CLIPModelPredictor.
        Args:
            model (HuggingFaceModel): The context of the model to be used.
            tags_list (List[str]): List of tags for inference.
            device (str): The device ('cpu' or 'gpu') on which to run the model.
        """
        super().__init__(model)
        if not hasattr(self.model, "loaded_processor"):
            raise ValueError("The model does not have a processor attribute.")
        self.tags_list = tags_list
        self.device = get_device(device)

    def pre_process_datalake(self, datalake: Datalake) -> tuple[list, list[str]]:
        """
        Pre-processes images from the datalake by converting them into inputs for the model.
        Args:
            datalake (Datalake): The context containing the directory of images.
        Returns:
            Tuple[List, List[str]]: A tuple containing the list of preprocessed inputs and image paths.
        """
        inputs = []
        image_paths = []
        for image_name in os.listdir(datalake.image_dir):
            image_path = os.path.join(datalake.image_dir, image_name)
            image_paths.append(image_path)
            image = Image.open(image_path)

            input = self.model.loaded_processor(
                images=image,
                text=self.tags_list,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            inputs.append(input)

        return inputs, image_paths

    def prepare_batches(self, images: list[Any], batch_size: int) -> list[list[str]]:
        """
        Splits the given images into batches of specified size.
        Args:
            images (List[Any]): A list of images to split into batches.
            batch_size (int): The size of each batch.
        Returns:
            List[List[str]]: A list of image batches.
        """
        return [images[i : i + batch_size] for i in range(0, len(images), batch_size)]

    def run_inference_on_batches(
        self, image_batches: list[list[str]]
    ) -> list[list[str]]:
        """
        Runs inference on each batch of images.
        Args:
            image_batches (List[List[str]]): List of image batches for inference.
        Returns:
            List[List[str]]: A list of predicted labels for each batch.
        """
        all_batch_results = []

        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_inputs=batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_inputs: list) -> list[str]:
        """
        Runs the model inference on a batch of inputs.
        Args:
            batch_inputs (List): A batch of pre-processed image inputs.
        Returns:
            List[str]: A list of predicted labels for the batch.
        """
        answers = []
        for input in batch_inputs:
            outputs = self.model.loaded_model(**input)
            probs = outputs.logits_per_image.softmax(dim=1)
            predicted_label = self.tags_list[probs.argmax().item()]
            answers.append(predicted_label)
        return answers

    def post_process_batches(
        self,
        image_batches: list[list[str]],
        batch_results: list[list[str]],
        datalake: Datalake,
    ) -> list[dict]:
        """
        Post-processes the batch predictions by mapping them to Picsellia tags and generating a final output.
        Args:
            image_batches (List[List[str]]): List of image batches.
            batch_results (List[List[str]]): List of batch prediction results.
            datalake (Datalake): The datalake for processing.
        Returns:
            List[Dict]: A list of dictionaries containing processed predictions.
        """
        all_predictions = []

        picsellia_tags_name = create_tags(
            datalake=datalake.datalake, list_tags=self.tags_list
        )

        for batch_result, batch_paths in zip(
            batch_results, image_batches, strict=False
        ):
            all_predictions.extend(
                self._post_process(
                    image_paths=batch_paths,
                    batch_prediction=batch_result,
                    datalake=datalake,
                    picsellia_tags_name=picsellia_tags_name,
                )
            )
        return all_predictions

    def _post_process(
        self,
        image_paths: list[str],
        batch_prediction: list[str],
        datalake: Datalake,
        picsellia_tags_name: dict[str, Tag],
    ) -> list[dict]:
        """
        Maps the predictions to Picsellia tags and returns processed predictions.
        Args:
            image_paths (List[str]): List of image paths.
            batch_prediction (List[str]): List of predictions for each image.
            datalake (Datalake): The datalake for retrieving data.
            picsellia_tags_name (Dict[str, Tag]): A dictionary of Picsellia tags.
        Returns:
            List[Dict]: A list of dictionaries containing data and their corresponding Picsellia tags.
        """
        processed_predictions = []

        for image_path, prediction in zip(image_paths, batch_prediction, strict=False):
            data_id = os.path.basename(image_path).split(".")[0]
            data = datalake.datalake.list_data(ids=[UUID(data_id)])[0]
            picsellia_tag = self.get_picsellia_tag(
                prediction=prediction, picsellia_tags_name=picsellia_tags_name
            )
            processed_prediction = {"data": data, "tag": picsellia_tag}
            processed_predictions.append(processed_prediction)

        return processed_predictions

    def get_picsellia_tag(
        self, prediction: str, picsellia_tags_name: dict[str, Tag]
    ) -> Tag:
        """
        Retrieves the Picsellia tag corresponding to the prediction.
        Args:
            prediction (str): The predicted tag name.
            picsellia_tags_name (Dict[str, Tag]): A dictionary mapping tag names to Tag objects.
        Returns:
            Tag: The corresponding Picsellia Tag object.
        Raises:
            ValueError: If the predicted tag is not found in Picsellia tags.
        """
        if prediction not in picsellia_tags_name:
            raise ValueError(f"Tag {prediction} not found in Picsellia tags.")
        return picsellia_tags_name[prediction]

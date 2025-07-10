from typing import Any

from picsellia import Label, ModelVersion
from picsellia_cv_engine.core.models import Model


class HuggingFaceModel(Model):
    """
    A context class specifically designed for managing HuggingFace models in the Picsellia platform.

    This class extends the general `Model` and adds additional functionalities
    to support HuggingFace models, including the ability to set and retrieve
    a processor object (such as a model or tokenizer) associated with the HuggingFace model.
    """

    def __init__(
        self,
        hugging_face_model_name: str,
        model_name: str,
        model_version: ModelVersion,
        pretrained_weights_name: str | None = None,
        trained_weights_name: str | None = None,
        config_name: str | None = None,
        exported_weights_name: str | None = None,
        labelmap: dict[str, Label] | None = None,
    ):
        """
        Initializes the `HuggingFaceModel` with model-related details.

        Args:
            hugging_face_model_name (str): The identifier of the HuggingFace model (e.g., 'bert-base-uncased').
            model_name (str): The name of the model as defined in Picsellia.
            model_version (ModelVersion): The specific version of the model within Picsellia.
            pretrained_weights_name (Optional[str]): The name of the pretrained weights file, if any.
            trained_weights_name (Optional[str]): The name of the trained weights file, if any.
            config_name (Optional[str]): The name of the configuration file for the model, if any.
            exported_weights_name (Optional[str]): The name of the exported weights file, if any.
            labelmap (Optional[Dict[str, Label]]): A dictionary mapping label names to `Label` objects in Picsellia.

        """
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            pretrained_weights_name=pretrained_weights_name,
            trained_weights_name=trained_weights_name,
            config_name=config_name,
            exported_weights_name=exported_weights_name,
            labelmap=labelmap,
        )
        self.hugging_face_model_name = hugging_face_model_name
        self._loaded_processor: Any | None = None

    @property
    def loaded_processor(self) -> Any:
        """
        Retrieves the processor currently loaded into the context.

        The processor can be a model, tokenizer, or any other relevant object needed
        for running inferences or processing tasks in HuggingFace models.

        Returns:
            Any: The processor object, typically a HuggingFace model or tokenizer.

        Raises:
            ValueError: If no processor has been set, raises an error indicating that the processor
            must be loaded before accessing it.
        """
        if self._loaded_processor is None:
            raise ValueError(
                "Processor is not loaded. Please set the model processor before accessing it."
            )
        return self._loaded_processor

    def set_loaded_processor(self, model: Any) -> None:
        """
        Assigns a processor to the context.

        The processor can be any model, tokenizer, or related object required for HuggingFace model
        operations. This method allows the user to specify which processor to use in the context.

        Args:
            model (Any): The processor to assign to the context, typically a HuggingFace model or tokenizer.
        """
        self._loaded_processor = model

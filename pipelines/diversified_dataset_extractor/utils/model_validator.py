import open_clip


def validate_pretrained_weights(
    pretrained_weights: str, available_pretrained_weights: str
) -> None:
    """
    Validate the provided pretrained weights.

    Args:
        pretrained_weights: The provided pretrained weights.
        available_pretrained_weights: The available pretrained weights.

    Raises:
        ValueError: If the provided pretrained weights are not available.
    """
    if pretrained_weights not in available_pretrained_weights:
        raise ValueError(
            f"The provided pretrained weights '{pretrained_weights}' are not available. "
            f"Available pretrained weights are {available_pretrained_weights}."
        )


def validate_model_architecture(
    model_architecture: str, available_model_names: str
) -> None:
    """
    Validate the provided model architecture.

    Args:
        model_architecture: The provided model architecture.
        available_model_names: The available model names.

    Raises:
        ValueError: If the provided model architecture is not available.
        NotImplementedError: If the provided model architecture is a HuggingFace model.
    """
    if model_architecture not in available_model_names:
        raise ValueError(
            f"The provided model '{model_architecture}' is not available. "
            f"Available models are {available_model_names}."
        )
    elif model_architecture.startswith(open_clip.factory.HF_HUB_PREFIX):
        raise NotImplementedError(
            f"The provided model '{model_architecture}' is a "
            f"HuggingFace model and is not supported yet. "
            f"Please provide a model from the list of available models: {available_model_names}."
        )

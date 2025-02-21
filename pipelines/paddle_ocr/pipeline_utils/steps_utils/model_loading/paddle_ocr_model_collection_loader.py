from pipelines.paddle_ocr.PaddleOCR.paddleocr import PaddleOCR


def paddle_ocr_load_model(
    bbox_model_path_to_load: str,
    text_model_path_to_load: str,
    character_dict_path_to_load: str,
    device: str,
) -> PaddleOCR:
    """
    Loads the PaddleOCR model using the provided model directories and character dictionary path.

    Args:
        bbox_model_path_to_load (str): The directory containing the bounding box detection model.
        text_model_path_to_load (str): The directory containing the text recognition model.
        character_dict_path_to_load (str): The path to the character dictionary file.
        device (str): The device on which the model should be loaded.

    Returns:
        PaddleOCR: The loaded PaddleOCR model.
    """

    if device.startswith("cuda") or device.startswith("gpu"):
        use_gpu = True
    else:
        use_gpu = False
    return PaddleOCR(
        use_angle_cls=True,
        rec_model_dir=text_model_path_to_load,
        det_model_dir=bbox_model_path_to_load,
        rec_char_dict_path=character_dict_path_to_load,
        use_gpu=use_gpu,
        show_log=False,
    )

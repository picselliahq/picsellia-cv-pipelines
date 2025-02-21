from typing import List

import numpy as np
import requests
from PIL import Image
from picsellia.sdk.asset import Asset
from torch import Tensor


def get_image_shape_with_exif_transpose(image: Image):
    """
        This method reads exif tags of an image and invert width and height if needed.
        Orientation flags that need inversion are : TRANSPOSE, ROTATE_90, TRANSVERSE and ROTATE_270

    Args:
        image: PIL Image to read

    Returns:
        width and height of image
    """
    exif = image.getexif()
    orientation = exif.get(0x0112)

    # Orientation when height and width are inverted :
    # 5: Image.Transpose.TRANSPOSE
    # 6: Image.Transpose.ROTATE_270
    # 7: Image.Transpose.TRANSVERSE
    # 8: Image.Transpose.ROTATE_90
    if orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 6:
        image = image.rotate(270, expand=True)
    elif orientation == 8:
        image = image.rotate(90, expand=True)
    return image


def is_labelmap_starting_at_zero(labelmap: dict) -> bool:
    return "0" in labelmap.keys()


def cast_type_list_to_int(_list: List) -> List:
    return list(map(int, _list))


def cast_type_list_to_float(_list: List) -> List:
    return list(map(float, _list))


def convert_tensor_to_list(tensor: Tensor) -> List:
    return tensor.cpu().numpy().tolist()


def rescale_normalized_box(box: List, width: int, height: int) -> List[int]:
    box = [
        box[0] * width,
        box[1] * height,
        (box[2] - box[0]) * width,
        (box[3] - box[1]) * height,
    ]
    return box


def open_asset_as_array(asset: Asset) -> np.array:
    image = Image.open(requests.get(asset.url, stream=True).raw)
    return np.array(image)

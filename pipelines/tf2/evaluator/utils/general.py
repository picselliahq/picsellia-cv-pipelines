from typing import List

import numpy as np
import requests
from picsellia.sdk.asset import Asset
from PIL import Image, ImageOps


def transpose_if_exif_tags(image: Image) -> Image:
    """Check if image needs to be transposed, and do the operation if so.

    This check is here to prevent unnecessary copy of the image by Pillow.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)
    if orientation != 1:
        image = ImageOps.exif_transpose(image)
    return image


def is_labelmap_starting_at_zero(labelmap: dict) -> bool:
    return "0" in labelmap.keys()


def cast_type_list_to_int(_list: List) -> List:
    return list(map(int, _list))


def cast_type_list_to_float(_list: List) -> List:
    return list(map(float, _list))


def convert_tensor_to_list(tensor) -> List:
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
    image = Image.open(requests.get(asset.reset_url(), stream=True).raw)
    image = transpose_if_exif_tags(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)

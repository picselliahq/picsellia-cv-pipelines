from typing import Optional

import numpy as np
import requests
import tensorflow as tf
from picsellia.sdk.asset import Asset
from PIL import Image


def open_asset_as_tensor(
    asset: Asset, input_width: Optional[int] = None, input_height: Optional[int] = None
):
    image = Image.open(requests.get(asset.reset_url(), stream=True).raw)
    if input_width is not None and input_height is not None:
        image = image.resize((input_width, input_height))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    if input_width is not None and input_height is not None:
        image = tf.convert_to_tensor(image, dtype=tf.float32)
    else:
        image = tf.convert_to_tensor(image, dtype=tf.uint8)

    return image

"""Compatibility shims for imgaug (NumPy 2, Python 3.14 multiprocessing).

imgaug 0.4.0 is unmaintained. Import this module before importing imgaug.

* It relies on NumPy aliases removed in NumPy 1.24 (np.bool, np.int, np.float,
  np.complex, np.object, np.str) and on np.sctypes, removed in NumPy 2.0.
* ``AddToHueAndSaturation`` builds a lookup table in ``__init__`` and stores it
  in a *class* attribute. Class attributes are not pickled, so DataLoader workers
  started with ``spawn``/``forkserver`` (the default on Linux since Python 3.14,
  they used to be forked) get ``None`` and crash. Build it lazily instead.
"""

import numpy as np

# Check ``vars(np)`` rather than ``hasattr``: accessing a removed alias such as
# ``np.object`` emits a FutureWarning on recent NumPy versions.
for _name, _target in (
    ("bool", np.bool_),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if _name not in vars(np):
        setattr(np, _name, _target)

if "sctypes" not in vars(np):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64, np.longdouble],
        "complex": [np.complex64, np.complex128, np.clongdouble],
        "others": [bool, object, bytes, str, np.void],
    }

from imgaug.augmenters.color import AddToHueAndSaturation as _AddToHueAndSaturation  # noqa: E402

_original_transform_image_cv2 = _AddToHueAndSaturation._transform_image_cv2.__func__


def _transform_image_cv2(cls, image_hsv, hue, saturation):
    if _AddToHueAndSaturation._LUT_CACHE is None:
        _AddToHueAndSaturation._LUT_CACHE = _AddToHueAndSaturation._generate_lut_table()
    return _original_transform_image_cv2(cls, image_hsv, hue, saturation)


_AddToHueAndSaturation._transform_image_cv2 = classmethod(_transform_image_cv2)

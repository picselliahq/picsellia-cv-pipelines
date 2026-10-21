"""NumPy compatibility shims for imgaug.

imgaug 0.4.0 is unmaintained and relies on NumPy aliases removed in NumPy 1.24
(np.bool, np.int, np.float, np.complex, np.object, np.str) and on np.sctypes,
removed in NumPy 2.0. Import this module before importing imgaug.
"""

import numpy as np

for _name, _target in (
    ("bool", np.bool_),
    ("int", int),
    ("float", float),
    ("complex", complex),
    ("object", object),
    ("str", str),
):
    if not hasattr(np, _name):
        setattr(np, _name, _target)

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64, np.longdouble],
        "complex": [np.complex64, np.complex128, np.clongdouble],
        "others": [bool, object, bytes, str, np.void],
    }

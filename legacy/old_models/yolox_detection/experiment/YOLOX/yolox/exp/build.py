#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file(exp_file, args):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp(args)
    except Exception:
        raise ImportError(f"{exp_file} doesn't contains class named 'Exp'")
    return exp


def get_exp_by_name(args):
    exp_name = args.name.replace("-", "_")
    module_name = ".".join(["YOLOX", "exps", "default", exp_name])
    exp_object = importlib.import_module(module_name).Exp(args)
    return exp_object


def get_exp(exp_file=None, exp_name=None, args=None):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
        args (Sequence[str]): the experiment arguments
    """
    assert exp_file is not None or exp_name is not None, (
        "plz provide exp file or exp name."
    )
    if exp_file is not None:
        return get_exp_by_file(exp_file, args)
    else:
        return get_exp_by_name(args)

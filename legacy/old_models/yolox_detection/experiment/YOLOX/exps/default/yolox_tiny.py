#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from YOLOX.yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

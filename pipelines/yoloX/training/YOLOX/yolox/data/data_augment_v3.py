#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import random

import cv2
import imgaug.augmenters as iaa
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from YOLOX.yolox.utils import xyxy2cxcywh
from loguru import logger


def mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransformV3:
    def __init__(self, enable_weather_transform: bool, max_labels: int = 50):
        self.max_labels = max_labels
        self.flip_prob = 0.5
        self.enable_weather_transform = enable_weather_transform
        self.aug = self._load_custom_augmentations()

    def _load_custom_augmentations(self):
        """Create a sequence of imgaug augmentations"""

        def sometimes(aug):
            return iaa.Sometimes(0.5, aug)

        def rarely(aug):
            return iaa.Sometimes(0.03, aug)

        def get_rare_transforms():
            transformations = [
                iaa.MotionBlur(k=15),
                iaa.OneOf(
                    [
                        # blur images with a sigma between 10 and 14.0
                        iaa.GaussianBlur((2, 4)),
                        # blur image using local means with kernel sizes
                        # between 2 and 5
                        iaa.AverageBlur(k=(2, 5)),
                        # blur image using local medians with kernel sizes
                        # between 1 and 3
                        iaa.MedianBlur(k=(1, 3)),
                    ]
                ),
            ]
            if self.enable_weather_transform:
                logger.info(
                    "Weather data augmentations (Cloud & Rain) have been enabled!"
                )
                transformations += [
                    iaa.CloudLayer(
                        intensity_mean=(220, 255),
                        intensity_freq_exponent=(-2.0, -1.5),
                        intensity_coarse_scale=2,
                        alpha_min=(0.7, 0.9),
                        alpha_multiplier=0.3,
                        alpha_size_px_max=(2, 8),
                        alpha_freq_exponent=(-4.0, -2.0),
                        sparsity=0.9,
                        density_multiplier=(0.3, 0.6),
                    ),
                    iaa.Rain(nb_iterations=1, drop_size=0.05, speed=0.2),
                ]

            return transformations

        return iaa.Sequential(
            [
                # execute 0 to 5 of the following (less important) augmenters per
                # image don't execute all of them, as that would often be way too
                # strong
                sometimes(
                    iaa.SomeOf(
                        (0, 5),
                        [
                            # convert images into their superpixel representation
                            iaa.Sharpen(
                                alpha=(0, 0.6), lightness=(0.9, 1.2)
                            ),  # sharpen images
                            # add gaussian noise to images
                            iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255)),
                            # change brightness of images (by -15 to 15 of original value)
                            iaa.AddToBrightness((-12, 12)),
                            iaa.AddToHueAndSaturation((-15, 15)),
                            iaa.Add((-8, 8), per_channel=0.5),
                        ],
                        random_order=True,
                    ),
                ),
                rarely(iaa.OneOf(get_rare_transforms())),
            ],
            random_order=True,
        )

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_aug = self.aug(image=image)
        image_t, boxes = mirror(image_aug, boxes, self.flip_prob)

        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes = boxes * r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels

    def visualize_image_with_boxes(self, image, boxes):
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for box in boxes:
            rect = patches.Rectangle(
                (box[0] - box[2] / 2, box[1] - box[3] / 2),
                box[2],
                box[3],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()


class ValTransformV3:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))

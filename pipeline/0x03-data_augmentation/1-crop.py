#!/usr/bin/env python3
"""Crop an image inside a tensor"""


import tensorflow as tf


def crop_image(image, size):
    """Crop an image inside a tensor"""
    return tf.image.random_crop(image, size)

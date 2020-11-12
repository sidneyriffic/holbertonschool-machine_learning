#!/usr/bin/env python3
"""Randomly shear an image tensor"""


import tensorflow as tf


def shear_image(image, intensity):
    """Randomly shear an image tensor"""
    return tf.keras.preprocessing.image.random_shear(image.numpy(), intensity,
                                                     channel_axis=2)

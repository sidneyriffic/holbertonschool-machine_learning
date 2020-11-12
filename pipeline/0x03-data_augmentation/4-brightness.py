#!/usr/bin/env python3
"""Randomly brighten an image."""


import tensorflow as tf


def change_brightness(image, max_delta):
    """Rotate a tensor image 90 degrees clockwise"""
    return tf.image.random_brightness(image, max_delta)

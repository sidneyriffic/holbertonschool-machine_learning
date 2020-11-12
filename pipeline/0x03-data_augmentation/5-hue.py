#!/usr/bin/env python3
"""Randomly brighten an image."""


import tensorflow as tf


def change_hue(image, delta):
    """Rotate a tensor image 90 degrees clockwise"""
    return tf.image.adjust_hue(image, delta)

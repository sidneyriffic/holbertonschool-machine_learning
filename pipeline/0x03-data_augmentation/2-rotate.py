#!/usr/bin/env python3
"""Rotate a tensor image 90 degrees clockwise"""

import tensorflow as tf


def rotate_image(image):
    """Rotate a tensor image 90 degrees clockwise"""
    return tf.image.rot90(image)

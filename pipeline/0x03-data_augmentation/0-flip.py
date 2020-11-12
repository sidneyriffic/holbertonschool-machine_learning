#!/usr/bin/env python3
"""Horizontal mirror an image inside a tensor"""


import tensorflow as tf


def flip_image(image):
    """Horizontal mirror an image inside a tensor"""
    return tf.image.flip_left_right(image)

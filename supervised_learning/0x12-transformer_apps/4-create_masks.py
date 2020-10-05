#!/usr/bin/env python3
"""Create masks for transformer"""


import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Create masks for transformer"""
    input_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    input_mask = input_mask[:, None, None, :]
    size = target.shape[1]
    band = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    band = tf.cast(band, tf.bool)[None, None, :, :]
    look_ahead_mask = tf.math.equal(target, 0)[:, None, None, :]
    look_ahead_mask = tf.cast(tf.math.logical_or(band, look_ahead_mask),
                              tf.float32)
    return input_mask, look_ahead_mask, input_mask

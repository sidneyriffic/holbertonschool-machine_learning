#!/usr/bin/env python3
"""Calculate scaled dot product attention"""


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot product attention"""
    denom = tf.math.sqrt(tf.cast(tf.shape(K)[-1], float))
    scaled = tf.matmul(Q, K, transpose_b=True) / denom
    if mask is not None:
        scaled = mask * -1e9 + scaled
    scaled = tf.nn.softmax(scaled, axis=-1)
    return tf.matmul(scaled, V), scaled

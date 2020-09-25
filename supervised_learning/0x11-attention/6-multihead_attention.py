#!/usr/bin/env python3
"""Calculate multi-head attention for a transformer"""


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot product attention"""

    denom = tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
    scaled = tf.matmul(Q, K, transpose_b=True) / denom
    if mask is not None:
        scaled = mask * -1e9 + scaled
    scaled = tf.nn.softmax(scaled, axis=-1)
    return tf.matmul(scaled, V), scaled


class MultiHeadAttention(tf.keras.layers.Layer):
    """Calculate multi-head attention for a transformer"""
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)
        self.depth = dm // h

    def call(self, Q, K, V, mask):
        """Keras layer call"""
        batches = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        def split_heads(x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        Q = split_heads(Q, batches)
        K = split_heads(K, batches)
        V = split_heads(V, batches)
        outs, weights = sdp_attention(Q, K, V, mask)
        outs = tf.transpose(outs, perm=[0, 2, 1, 3])
        outs = tf.reshape(outs, [batches, -1, self.dm])
        return self.linear(outs), weights

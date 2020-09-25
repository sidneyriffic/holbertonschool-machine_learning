#!/usr/bin/env python3
"""Transformer encoder block"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Keras layer call"""
        mid, _ = self.mha(x, x, x, mask)
        mid = self.dropout1(mid, training=training)
        mid = self.layernorm1(x + mid)
        out = self.dense_hidden(mid)
        out = self.dense_output(out)
        out = self.dropout2(out, training=training)
        out = self.layernorm2(mid + out)
        return out

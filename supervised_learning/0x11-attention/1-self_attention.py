#!/usr/bin/env python3
"""Create self-attention layer"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Create self-attention layer"""
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Call function for the layer"""
        energy = self.W(s_prev)[:, None, :] + self.U(hidden_states)

        return tf.math.reduce_sum(energy, axis=1), self.V(tf.math.tanh(energy))

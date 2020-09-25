#!/usr/bin/env python3
"""Make an RNN based encoder for NLP"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """An RNN encoder layer for NLP"""
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Initialize hidden state to all 0"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Keras call for the layer"""
        hidden = self.embedding(x)
        outs = self.gru(hidden, initial_state=initial)
        return outs

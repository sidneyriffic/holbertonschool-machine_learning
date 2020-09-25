#!/usr/bin/env python3
"""Transformer decoder"""


import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder"""
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.positional_encoding = positional_encoding(max_seq_len, dm)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Keras layer call"""
        return encoder_output

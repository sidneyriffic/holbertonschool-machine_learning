#!/usr/bin/env python3
"""Full transformer network"""


import tensorflow.compat.v2 as tf
import numpy as np


class Transformer(tf.keras.layers.Layer):
    """Transformer decoder"""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate=0.1)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate=0.1)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """Keras layer call"""
        enc_out = self.encoder(inputs, training, encoder_mask)
        dec_out = self.decoder(target, enc_out, training, look_ahead_mask,
                               decoder_mask)
        return self.linear(dec_out)


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
        seq_len = x.shape[1]

        out = self.embedding(x)
        out *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        out += self.positional_encoding[:seq_len, :]
        out = self.dropout(out, training=training)

        for i in range(self.N):
            out = self.blocks[i](out, encoder_output, training,
                                 look_ahead_mask, padding_mask)
        return out


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder"""
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.positional_encoding = positional_encoding(max_seq_len, dm)

    def call(self, x, training, mask):
        """Keras layer call"""
        seq_len = x.shape[1]
        out = self.embedding(x)
        out *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        out += self.positional_encoding[None, :seq_len, :]
        out = self.dropout(out, training=training)
        for i in range(self.N):
            out = self.blocks[i](out, training, mask)
        return out

    
class DecoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """Keras layer call"""
        start, weights1 = self.mha1(x, x, x, look_ahead_mask)
        start = self.dropout1(start, training=training)
        start = self.layernorm1(x + start)
        mid, weights2 = self.mha2(start, encoder_output, encoder_output,
                                  padding_mask)
        mid = self.dropout2(mid, training=training)
        mid = self.layernorm2(start + mid)
        out = self.dense_hidden(mid)
        out = self.dense_output(out)
        out = self.dropout3(out, training=training)
        out = self.layernorm3(mid + out)
        return out


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


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot product attention"""
    denom = tf.math.sqrt(tf.cast(tf.shape(K)[-1], float))
    scaled = tf.matmul(Q, K, transpose_b=True) / denom
    if mask is not None:
        scaled = mask * -1e9 + scaled
    scaled = tf.nn.softmax(scaled, axis=-1)
    return tf.matmul(scaled, V), scaled


def positional_encoding(max_seq_len, dm):
    """Calculate positional encoding for a transformer"""
    pos_enc = np.ndarray((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(dm):
            if j % 2:
                pos_enc[i][j] = np.cos(i / np.power(10000, (j - 1) / dm))
            else:
                pos_enc[i][j] = np.sin(i / np.power(10000, j / dm))
    return pos_enc

#!/usr/bin/env python3
"""Final. Set up dataset pipeline for fitting."""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """Portuguese to English dataset"""
    def __init__(self, batch_size, max_len):
        print("initting")
        tf.compat.v1.enable_eager_execution()
        print("Execute tokenizing eagerly?", tf.executing_eagerly())
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True).take(20)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True).take(20)
        """
        def build_numpy(dataset):
            list = []
            for pt, en in dataset:
                list.append([pt.numpy(), en.numpy()])
            return np.asarray(list)
        self.data_train = build_numpy(self.data_train)
        #self.data_valid = build_numpy(self.data_train)
        print(self.data_train)
        """
        pt, en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = pt
        self.tokenizer_en = en
        """
        print(self.data_train[:, 0, None])
        def encode(encoder, data):
            list = []
            for sentence in data:
                list.append(encoder.encode(sentence))
            return np.asarray(list)
        self.data_train[:, 0] = np.apply_along_axis(pt.encode, 1, self.data_train[:, 0, None])
        self.data_train[:, 1] = np.apply_along_axis(en.encode, 1, self.data_train[:, 1, None])
        #self.data_train = tf.data.Dataset.from_generator(self.tf_encode, (tf.int64, tf.int64), args=[(x, y) for x, y in self.data_train])
        print(self.data_train)
        #self.data_valid = tf.data.Dataset.from_generator(self.tf_encode, (tf.int64, tf.int64), args=[(x, y) for x, y in self.data_valid])
        exit()
        """

        tf.compat.v1.disable_eager_execution()
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True).take(20)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True).take(20)
        print("Execute preprocessing eagerly?", tf.executing_eagerly())
        print(self.data_train)
        self.data_train = self.data_train.map(lambda x, y:
                                              tf.py_function(self.tf_encode,
                                                             [x, y],
                                                             (tf.int64,
                                                              tf.int64)))
        self.data_valid = self.data_valid.map(lambda x, y:
                                              tf.py_function(self.tf_encode,
                                                             [x, y],
                                                             (tf.int64,
                                                              tf.int64)))
        self.data_train = tf.data.Dataset.from_tensors(self.data_train)
        print("new data", self.data_train)
        def test(x):
            print("in test", x)
            return True
        self.data_train = self.data_train.filter(lambda x: test(x))
        #self.data_valid = self.data_valid.filter(lambda x: test(x))
        self.data_train = self.data_train.cache().shuffle(10000000)
        #self.data_train = self.data_train.padded_batch(batch_size,
        #                                               ([None], [None]))
        #self.data_train = self.data_train.prefetch(tf.data.experimental.
        #                                           AUTOTUNE)
        #self.data_valid = self.data_valid.padded_batch(batch_size,
        #                                               ([None], [None]))


    def tokenize_dataset(self, data):
        """Create subword tokenizers"""
        builder = tfds.features.text.SubwordTextEncoder.build_from_corpus
        pt = builder((pt.numpy() for pt, _ in data.repeat(1)), 2**15)
        en = builder((en.numpy() for _, en in data.repeat(1)), 2**15)
        return pt, en

    def encode(self, pt, en):
        """Encode a translation pair into tokens"""
        pt = self.tokenizer_pt.encode(pt.numpy())
        en = self.tokenizer_en.encode(en.numpy())
        vocab_size = self.tokenizer_pt.vocab_size
        pt.insert(0, vocab_size)
        pt.append(vocab_size + 1)
        vocab_size = self.tokenizer_en.vocab_size
        en.insert(0, vocab_size)
        en.append(vocab_size + 1)
        return pt, en

    def tf_encode(self, pt, en):
        """Tensorflow dataset wrapper for encoding"""
        pt, en = self.encode(pt, en)
        pt = tf.cast(pt, tf.int64)
        en = tf.cast(en, tf.int64)
        return tf.convert_to_tensor(pt), tf.convert_to_tensor(en)

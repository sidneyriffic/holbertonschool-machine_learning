#!/usr/bin/env python3
"""Encode tokens from Portuguese to English dataset"""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Portuguese to English dataset"""
    def __init__(self):
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        pt, en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = pt
        self.tokenizer_en = en

    def tokenize_dataset(self, data):
        """Create subword tokenizers"""
        tf.compat.v1.enable_eager_execution()
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

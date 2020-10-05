#!/usr/bin/env python3
"""Set up dataset for training"""


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

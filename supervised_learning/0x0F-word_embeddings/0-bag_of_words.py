#!/usr/bin/env python3
"""Extract bag of words representation"""


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    data = vectorizer.fit_transform(sentences)
    return data.toarray(), vectorizer.get_feature_names()

#!/usr/bin/env python3
"""Extract tf_idf representation"""


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    data = vectorizer.fit_transform(sentences)
    return data.toarray(), vectorizer.get_feature_names()


def tf_idf(sentences, vocab=None):
    """Extract tf_idf representation"""
    bow, vocab = bag_of_words(sentences, vocab)
    return TfidfTransformer().fit_transform(bow).toarray(), vocab

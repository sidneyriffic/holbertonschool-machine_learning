#!/usr/bin/env python3
"""Put a gensim word2vec model in a keras embedding layer"""


from tensorflow import keras


def gensim_to_keras(model):
    """Put a gensim word2vec model in a keras embedding layer"""
    return model.wv.get_keras_embedding(train_embeddings=True)

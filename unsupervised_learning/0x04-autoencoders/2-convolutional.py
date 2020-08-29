#!/usr/bin/env python3
"""A sparse autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """A sparse autoencoder"""
    encodein = keras.Input(shape=input_dims)
    latents = encodein
    for layer in filters:
        latents = keras.layers.Conv2D(layer, 3, activation="relu",
                                      padding="same")(latents)
        latents = keras.layers.MaxPool2D(padding="same")(latents)
    encoder = keras.Model(encodein, latents)
    encoder.summary()
    decodein = keras.Input(shape=latent_dims)
    recons = decodein
    for layer in filters[::-1][:-1]:
        recons = keras.layers.Conv2D(layer, 3, activation="relu",
                                     padding="same")(recons)
        recons = keras.layers.UpSampling2D()(recons)
    if len(filters):
        recons = keras.layers.Conv2D(filters[0], 3, activation="relu")(recons)
        recons = keras.layers.UpSampling2D()(recons)
    recons = keras.layers.Conv2D(input_dims[2], input_dims[0:2],
                                 activation="sigmoid", padding="same")(recons)
    decoder = keras.Model(decodein, recons)
    decoder.summary()
    autoencoder = keras.Model(encodein, decoder(latents))
    autoencoder.summary()
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, autoencoder

#!/usr/bin/env python3
"""A vanilla autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """A vanilla autoencoder"""
    encodein = keras.Input(shape=(input_dims,))
    latents = encodein
    for layer in hidden_layers:
        latents = keras.layers.Dense(layer, activation="relu")(latents)
    latents = keras.layers.Dense(latent_dims, activation="relu")(latents)
    encoder = keras.Model(encodein, latents)
    decodein = keras.Input(shape=(latent_dims,))
    recons = decodein
    for layer in hidden_layers[::-1]:
        recons = keras.layers.Dense(layer, activation="relu")(recons)
    recons = keras.layers.Dense(input_dims, activation="sigmoid")(recons)
    decoder = keras.Model(decodein, recons)
    autoencoder = keras.Model(encodein, decoder(latents))
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, autoencoder

#!/usr/bin/env python3
"""A sparse autoencoder"""


import tensorflow.keras as keras


def sample(args):
    """Sample from variational space for output"""
    mean, logvar = args
    batch = keras.backend.shape(mean)[0]
    dim = keras.backend.int_shape(mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return mean + keras.backend.exp(0.5 * logvar) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """A variational autoencoder"""
    encodein = keras.Input(shape=(input_dims,))
    middle = encodein
    for layer in hidden_layers:
        middle = keras.layers.Dense(layer, activation="relu")(middle)
    mean = keras.layers.Dense(latent_dims)(middle)
    logvar = keras.layers.Dense(latent_dims)(middle)
    latents = keras.layers.Lambda(sample,
                                  output_shape=latent_dims)([mean, logvar])
    encoder = keras.Model(encodein, [mean, logvar, latents])
    decodein = keras.Input(shape=(latent_dims,))
    recons = decodein
    for layer in hidden_layers[::-1]:
        recons = keras.layers.Dense(layer, activation="relu")(recons)
    recons = keras.layers.Dense(input_dims, activation="sigmoid")(recons)
    decoder = keras.Model(decodein, recons)
    output = decoder(encoder(encodein)[2])
    autoencoder = keras.Model(encodein, output)
    recons_loss = keras.losses.binary_crossentropy(encodein, output)
    recons_loss *= input_dims
    kl_loss = 1 + logvar - keras.backend.square(mean)
    kl_loss -= keras.backend.exp(logvar)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss /= -2
    vae_loss = keras.backend.mean(recons_loss + kl_loss)
    autoencoder.add_loss(vae_loss)
    encoder.summary()
    decoder.summary()
    autoencoder.summary()
    autoencoder.compile(optimizer="adam", loss="mse")
    return encoder, decoder, autoencoder

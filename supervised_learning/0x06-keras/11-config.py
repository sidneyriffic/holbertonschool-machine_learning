#!/usr/bin/env python3
"""Save and load a whole keras model"""


import tensorflow.keras as K


def save_config(network, filename):
    """Save a whole keras model"""
    with open(filename, 'w+') as file:
        file.write(network.to_json())


def load_config(filename):
    """Load a whole keras model"""
    with open(filename, 'r') as file:
        return K.models.model_from_json(file.read())

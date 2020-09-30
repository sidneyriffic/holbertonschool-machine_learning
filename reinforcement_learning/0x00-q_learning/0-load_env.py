#!/usr/bin/env python3
"""Load the frozen lake environment from Open AI's gym"""


import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the frozen lake environment from Open AI's gym"""
    return gym.make('FrozenLake-v0', desc=desc,
                    map_name=map_name, is_slippery=is_slippery)

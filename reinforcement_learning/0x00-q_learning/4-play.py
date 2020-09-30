#!/usr/bin/env python3
"""Have the agent play a game of FrozenLake using what it learned"""


epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def play(env, Q, max_steps=100):
    """Have the agent play a game of FrozenLake using what it learned"""
    state = 0
    env.reset()
    env.render()
    for step in range(max_steps):
        act = epsilon_greedy(Q, state, 0)
        state, reward, done, _ = env.step(act)
        env.render()
        if done and reward == 0:
            return reward
        if(done):
            return reward

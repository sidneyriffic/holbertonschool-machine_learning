#!/usr/bin/env python3
"""Train reinforcement model using Q-learning"""


import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Train reinforcement model using Q-learning"""
    total_rewards = []
    for episode in range(episodes):
        env.reset()
        prev_state = 0
        reward = 0
        state = 0
        for step in range(max_steps):
            act = epsilon_greedy(Q, state, epsilon)
            state, reward, done, _ = env.step(act)
            if done and reward == 0:
                reward = -1
            Q[prev_state, act] += (alpha * (reward + gamma
                                            * np.max(Q[state])
                                            - Q[prev_state, act]))
            prev_state = state
            if(done):
                break
        epsilon = max(epsilon * (1-epsilon_decay), min_epsilon)
        total_rewards.append(reward)
    return Q, total_rewards

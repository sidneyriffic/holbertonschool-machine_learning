#!/usr/bin/env python3
"""Train TD(Lambda) value estimation on a OpenAI gym"""


import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=.99):
    """Train TD(Lambda) value estimation on a OpenAI gym"""
    success = 0
    for episode in range(episodes):
        # print("episode", episode)
        prev_state = env.reset()
        eligibility = np.zeros((len(V),))
        reward = 0
        # env.render()
        for step in range(max_steps):
            action = policy(prev_state)
            state, reward, done, info = env.step(action)
            eligibility *= gamma * lambtha
            eligibility[prev_state] += 1
            # print(eligibility.reshape(8, 8))
            # To prevent incorrect terminating values from messing with
            # valuations we'll assume the value of termination is equal to
            # the reward given at termination time.
            if done:
                td_error = reward - V[prev_state]
                print(eligibility.reshape((8,8)))
            else:
                td_error = reward + gamma * V[state] - V[prev_state]
            V = V + alpha * td_error * eligibility
            prev_state = state
            # env.render()
            if reward > 0:
                success += 1
            if done:
                break
    print("successes", success)
    return V

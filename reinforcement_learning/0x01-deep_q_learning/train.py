#!/usr/bin/env python3
"""Train a reinforcement policy to play Breakout"""


import rl
import gym
import keras
import numpy as np
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger


INPUT_SHAPE = (84, 84)


# Processor structure used from the Keras RL example.
class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`.
        # In this case, however, we would need to store a `float32` array
        # instead, which is 4x more memory intensive than an `uint8` array.
        # This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                              value_min=.1, value_test=0.05, nb_steps=1000000)
memory = SequentialMemory(limit=1000000, window_length=4)


input = keras.layers.Input((4, 84, 84))
model = keras.layers.Permute((2, 3, 1))(input)


model = keras.layers.Conv2D(32, (8, 8), strides=(4, 4),
                            activation='relu')(model)
model = keras.layers.Conv2D(64, (4, 4), strides=(2, 2),
                            activation='relu')(model)
model = keras.layers.Conv2D(64, (3, 3), activation='relu')(model)
model = keras.layers.Flatten()(model)
model = keras.layers.Dense(512, activation='relu')(model)
model = keras.layers.Dense(4, activation='linear')(model)


model = keras.Model(inputs=input, outputs=model)
model.summary()


print(model.output)
model.output._keras_shape = (None, 4)
print(model.output._keras_shape)
game = gym.make('Breakout-v0')
agent = DQNAgent(model, policy, nb_actions=game.action_space.n,
                 nb_steps_warmup=50000, memory=memory,
                 processor=AtariProcessor(), train_interval=4, delta_clip=1.)
agent.compile(keras.optimizers.Adam(lr=.00025), metrics=['mae'])

callbacks = [rl.callbacks.ModelIntervalCheckpoint('ckpt.h5f', interval=250000)]
callbacks += [FileLogger('log.json', interval=100)]
if False:
    agent.load_weights('weights.h5f')
agent.fit(game, nb_steps=1750000, visualize=False, log_interval=10000,
          callbacks=callbacks)
agent.save_weights('weights.h5f', overwrite=True)
game.reset()
agent.test(game, nb_episodes=10, visualize=True)

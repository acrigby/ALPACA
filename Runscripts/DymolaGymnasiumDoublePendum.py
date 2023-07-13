# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:33:41 2023

@author: localuser
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt

from dymola.dymola_interface import DymolaInterface

model = "DoublePendulum"

dymola = DymolaInterface()
print(dymola.DymolaVersion())
dymola.openModel(model)

learningtimes = []
learningtimesav = []
iterations = []
window = 5
i = 0

env = gym.make('AcrobotDymola', render_mode = "human")
#env = TimeLimit(env, max_episode_steps=5000)
observation, info = env.reset()


for _ in range(1000000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        i += 1
        plt.plot(env.results["Time"],env.results["revolute1.phi"])
        plt.plot(env.results["Time"],env.results["revolute2.phi"])
        plt.show()
        print(env.t)
        learningtimes.append(env.t)
        iterations.append(i)
        if i < window:
            learningtimesav.insert(0,np.nan)
        else:
            learningtimesav.append(np.mean(learningtimes[i-window:i]))

        plt.plot(iterations,learningtimes, 'k.-', label='Original data')
        plt.plot(iterations,learningtimesav, 'r.-', label='Running average')
        plt.yscale('log')
        plt.grid(linestyle=':')
        plt.legend()
        plt.show()
        observation, info = env.reset()

env.close()

plt.plot(learningtimes, 'k.-', label='Original data')
plt.plot(learningtimesav, 'r.-', label='Running average')
plt.show()
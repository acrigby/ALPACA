# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:33:41 2023

@author: localuser
"""

import gymnasium as gym
#from gymnasium.wrappers import TimeLimit
import numpy as np
import matplotlib.pyplot as plt

#model = "DoublePendulum"


learningtimes = []
learningtimesav = []
iterations = []
window = 5
t = 0
i = 0

env = gym.make('Acrobot-v1', render_mode = 'human')
#env = TimeLimit(env, max_episode_steps=5000)
observation, info = env.reset()


for _ in range(10000):
    t += 0.2
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        i += 1
        print(t)
        learningtimes.append(t)
        iterations.append(i)
        if i < window:
            learningtimesav.insert(0,np.nan)
        else:
            learningtimesav.append(np.mean(learningtimes[i-window:i]))
        z = np.polyfit(iterations,learningtimes, 1)
        p = np.poly1d(z)
        
        #add trendline to plot
        plt.plot(iterations, p(iterations), label = 'Trendline')
        plt.plot(iterations,learningtimes, 'k.-', label='Original data')
        plt.plot(iterations,learningtimesav, 'r.-', label='Running average')
        plt.yscale('log')
        plt.grid(linestyle=':')
        plt.legend()
        plt.savefig('C:/Users/aidan/projects/ALPACA/Figures/learning.png')
        plt.show()
        plt.plot(iterations, p(iterations), label = 'Trendline')
        plt.plot(iterations,learningtimes, 'k.-', label='Original data')
        plt.plot(iterations,learningtimesav, 'r.-', label='Running average')
        plt.grid(linestyle=':')
        plt.legend()
        plt.savefig('C:/Users/aidan/projects/ALPACA/Figures/learning.png')
        plt.show()
        t = 0
        observation, info = env.reset()

env.close()

plt.plot(learningtimes, 'k.-', label='Original data')
plt.plot(learningtimesav, 'r.-', label='Running average')
plt.show()
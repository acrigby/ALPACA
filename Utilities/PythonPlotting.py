# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 08:38:03 2023

@author: localuser
"""

import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('MatlabConvertedFile.mat')

time = mat['coldTemp'][:,0]
coldTemp = mat['coldTemp'][:,1] - 273.15
hotTemp = mat['hotTemp'][:,1] - 273.15
plt.plot(time,coldTemp)
plt.plot(time,hotTemp)
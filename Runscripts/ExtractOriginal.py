# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:01:11 2023

@author: localuser
"""

from dymola.dymola_interface import DymolaInterface
import matplotlib.pyplot as plt
import numpy as np

dymola = DymolaInterface()
print(dymola.DymolaVersion())

results = {}
variables = ["Time","sensor_pT.T"]    
for key in variables:
    results[key] = []
trajsize = dymola.readTrajectorySize("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/Original_Temp_Profile.mat")
signals = dymola.readTrajectory("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/Original_Temp_Profile.mat", variables, trajsize)

for i in range(0,len(variables),1):
    results[variables[i]].extend(signals[i])
    
for i in range(0,len(results["Time"]),1):
    results["Time"][i] =  results["Time"][i] - 9900
    
plt.plot(results["Time"],results["sensor_pT.T"])
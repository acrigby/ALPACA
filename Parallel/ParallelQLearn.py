# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:53:02 2023

@author: localuser
"""

import subprocess
from DyMat.DyMat import DyMatFile
import matplotlib.pyplot as plt





subprocess.check_call("C:/Users/localuser/Documents/GitHub/ALPACA/Parallel/dymosim")

r = DyMatFile("dsres.mat")


results = {}
variables = ["BOP.sensor_pT.T","BOP.sensor_pT.p","BOP.sensor_T2.T","BOP.TCV.dp","pump_SimpleMassFlow1.m_flow", "ramp.y", "boundary.Q_flow_ext","BOP.sensorW.W","FeedForward.y"]
t = r.abscissa(variables[0])[0]
Pout = r.data("BOP.sensor_pT.p")

plt.plot(t, Pout)
# Tin = results["BOP.sensor_T2.T"][-1]
# TCVdp = results["BOP.TCV.dp"][-1]
# PumpMFlow = results["pump_SimpleMassFlow1.m_flow"][-1]
# Qdemand = results["ramp.y"][-1]
# Qout = results["BOP.sensorW.W"][-1]
# FF = results["FeedForward.y"][-1]
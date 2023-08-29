from dymola.dymola_interface import DymolaInterface
import matplotlib.pyplot as plt
import numpy as np
import os

#Path to dymola output directory of your choosing
WD = "C:/Users/localuser/Documents/Dymola"


# Instantiate the Dymola interface and start Dymola
dymola = DymolaInterface()
print(dymola.DymolaVersion())

#open HYBRID with path to your HYBRID Library
dymola.openModel("C:/Users/localuser/HYBRID/Models/NHES/package.mo")
#open TRANSFORM
dymola.openModel("C:/Users/localuser/HYBRID/TRANSFORM-Library/TRANSFORM-Library/TRANSFORM/package.mo")


#String of the model you wante to simulate
model = "NHES.Systems.BalanceOfPlant.Turbine.Examples.SteamTurbine_L1_boundaries_Test_a" 

#Modelica String Paths of the variables you want to investigate - always leave "Time" as first variable
variables = ["Time","BOP.sensorW.W"]

#Open the model of interest
dymola.openModel(model)

#Set the Dymola working directory to where you want your dymola output
#This command just executes the string given in the dymola terminal so make sure you use speech marks not just apostrophes
dymola.ExecuteCommand('Modelica.Utilities.System.setWorkDirectory("'+ WD +'")') 

#Simulate the model - can specify start and end and either number of intervals of output interval (set other one to zero). Specify name of results file
result = dymola.simulateModel(model, startTime=0, stopTime=300, numberOfIntervals=0, outputInterval=1, method="Esdirk45a",resultFile="PythonDymola");

if not result:
    print("Simulation failed. Below is the translation log.")
    log = dymola.getLastErrorLog()
    print(log)
    dymola.exit(1)
    
#Make a Python dictionary to hold all the results base on variables in variables list
results = {}
for key in variables:
         results[key] = []

#read the .Mat file's size and then write the variables to a list
trajsize = dymola.readTrajectorySize(WD + "/PythonDymola.mat")
signals=dymola.readTrajectory(WD + "/PythonDymola.mat", variables, trajsize)

#add the result signals to the results dictonary
for i in range(0,len(variables),1):
    results[variables[i]].extend(signals[i])

#plot the power
fig, ax1 = plt.subplots(figsize=(9, 6))
ax1.plot(results["Time"],results["BOP.sensorW.W"],  'r.-')
ax1.grid(linestyle=':')
ax1.set_xlabel("Time / s")
ax1.set_ylabel("BOP Power / W")
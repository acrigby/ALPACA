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
variables = ["Time","BOP.sensorW.W","BOP.CS.TCV_opening_nominal", "BOP.valve_TCV.m_flow"]

#Open the model of interest
dymola.openModel(model)

#Set the Dymola working directory to where you want your dymola output
#This command just executes the string given in the dymola terminal so make sure you use speech marks not just apostrophes
dymola.ExecuteCommand('Modelica.Utilities.System.setWorkDirectory("'+ WD +'")') 

#define a logarithmic spacing of turbine control valve positions
openings = np.logspace(-2,0,10)

#create empty lists to hold average values
powers = []
mflows = []

#generate dictionary to hold results
results = {}
for key in variables:
    results[key] = []
 
#define timestepping interval
timeinterval = 5
    
#loop over time steps
for t_end in range(0,300,timeinterval):
    
    #define condensor pressure
    p_condensor = 1e4 + 9e4*(t_end/300)
    #change an input variable - ensure it can be changed by running a test command in the dymola command line
    var = "BOP.p_condenser ="+ str(p_condensor)
    print(var)
    dymola.ExecuteCommand(var)
    
    #define start and end of simulation
    start = t_end
    stop = t_end+timeinterval

    #simulate one time step
    result = dymola.simulateModel(model, startTime=start, stopTime=stop, numberOfIntervals=0, outputInterval=0.1, method="Esdirk45a", resultFile="PythonDymola")

    #print log if error
    if not result:
        print("Simulation failed. Below is the translation log.")
        log = dymola.getLastErrorLog()
        print(log)
        dymola.exit(1)


    #get results and add to dictionary
    trajsize = dymola.readTrajectorySize(WD + "/PythonDymola.mat")
    signals=dymola.readTrajectory(WD + "/PythonDymola.mat", variables, trajsize)

    for i in range(0,len(variables),1):
        results[variables[i]].extend(signals[i])
        
    #import final conditions of previous timestep prior to starting the new one
    dymola.ExecuteCommand('importInitial("'+ WD + '/dsfinal.txt")')


#plot episode trajectory        
plt.plot(results["Time"],results["BOP.sensorW.W"], label = "Power Output")
plt.grid(linestyle=':')
plt.xlabel("Time / s")
plt.ylabel("Power / W")
plt.show()


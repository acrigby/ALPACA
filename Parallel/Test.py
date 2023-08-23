# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:07:10 2023

@author: localuser
"""

import pathlib
import os
from dymola.dymola_interface import DymolaInterface

print(os.path.abspath(os.curdir))
os.chdir("..")
print(os.path.abspath(os.curdir))
ALPACApath = os.path.abspath(os.curdir)

# Instantiate the Dymola interface and start Dymola
dymola = DymolaInterface()
print(dymola.DymolaVersion())

#Define Model File
model = 'ControlTests.SteamTurbine_L2_OpenFeedHeat_Test2'
 
# #Define Previous working directory
# dymola.AddModelicaPath("C:/Users/localuser/Documents/Dymola")

#open the dymola model in the environment 
# dymola.openModel(model)

#open modelica paths file and open modelica paths
pathsfile = open(str(ALPACApath)+'/Utilities/ModelicaPaths.txt', 'r')
Paths = pathsfile.readlines()
for path in Paths:
    path = path.strip()
    dymola.openModel(path)

#Add any package dependencies to the enviroment and change working directory

dymola.openModel(str(ALPACApath)+"/ModelicaFiles/ControlTests.mo")
wd = 'Modelica.Utilities.System.setWorkDirectory("' + str(ALPACApath) + '\Runscripts")'
print(wd)
dymola.ExecuteCommand(wd) 

#reopen model in new working directory
dymola.openModel(model)

#translate the model and create dymosym files
dymola.translateModel(model)

#Initialise the model and create first dsfinal file
result = dymola.simulateModel(model, startTime=0, stopTime = 9900, outputInterval=100, method="Esdirk45a",resultFile="SteamTurbine_L2_OpenFeedHeat_Test2")
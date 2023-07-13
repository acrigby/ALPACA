from dymola.dymola_interface import DymolaInterface
import os

# Instantiate the Dymola interface and start Dymola
dymola = DymolaInterface()

# Call a function in Dymola and check its return value
result = dymola.simulateModel("Modelica.Mechanics.Rotational.Examples.CoupledClutches")
if not result:
    print("Simulation failed. Below is the translation log.")
    log = dymola.getLastErrorLog()
    print(log)
    dymola.exit(1)

dymola.plot(["J1.w", "J2.w", "J3.w", "J4.w"])
plotPath = os.getcwd() + "/plot.png"
dymola.ExportPlotAsImage(plotPath)

print("OK")

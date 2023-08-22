"""classic Acrobot task"""
from typing import Optional

import numpy as np
from numpy import cos, pi, sin

from dymola.dymola_interface import DymolaInterface

import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

"""Adapted from the acrobot enviroment in Gymnasium by Aidan Rigby aidan.rigby@inl.gov to perform Dymola Optimisation"""

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class ReactorEnv(Env):
    """
    ## Description

    This enviorment interfaces with the L2_boundaries balance of plant model found in the HYBRID library of 
    modelica models developped at Idaho National Laboratory: https://github.com/idaholab/HYBRID. The aim is 
    to find an optimised feedforward signal to provide to the feedwater coolant pump to minimise the temperature
    deviation at the steam generator exit.
    
    Details of this code can be found at {osti link}

    ## Action Space

    The action is discrete, deterministic, and represents the change in the feedforward signal applied on the FWCP.

    | Num | Action                                | Unit         |
    |-----|---------------------------------------|--------------|
    | 0   | change the feedforward signal by -0.7 | mass flow rate (kg/s) |
    | 1   | change the feedforward signal by -0.2 | mass flow rate (kg/s) |
    | 2   | change the feedforward signal by -0.1 | mass flow rate (kg/s) |
    | 3   | change the feedforward signal by  0   | mass flow rate (kg/s) |
    | 4   | change the feedforward signal by  0.1 | mass flow rate (kg/s) |
    | 5   | change the feedforward signal by  0.2 | mass flow rate (kg/s) |
    | 6   | change the feedforward signal by  0.7 | mass flow rate (kg/s) |

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` that provides information about the
    steam generator outlet temperature and power demand as well as current pump states. The states are normalised 
    to aid initialisation:

    | Num | Observation                        |Normalization       | Min                 | Max               |
    |-----|------------------------------------|--------------------|---------------------|-------------------|
    | 0   | SG Outlet Temperature              |   T_out - 673.1    | -8                  | 8                 |
    | 1   | FWCP mass flow rate                |    m_pump-58       | -10                 | 10                |
    | 2   | Turbine Electrical Power Output    | Q_e - 30 MW / 1 MW | -6                  |  6                | 
    | 3   | Pump Controller Feedforward Signal |      N/A           | -10                 | 10                |

    ## Rewards

	The reward takes the form of a scalar variable that adds a contribution to the score at each time step 
    dependent on how close the output temperature is at the end of that time step to the goal condition of 400°C. 
    This takes the form of a linear function described by equation 1.
    █(R[t]= 8-|〖(T〗_out [t]-673.15)|#1)


    ## Starting State

    The starting state is initialised using a reset file in the host repository of Starting_9900.txt

    ## Episode End

    The episode is terminated, and the score returned under two conditions. 
    1. If the SG output temperature at the end of any given time step does not lie in the 
    range: 664.15 < T_out < 682.15. This reduces the time the simulation takes to fully learn the sub space 
    by penalizing bad states. 
    2. The simulation has a run time set to 100 steps. After this the change in the 
    feedforward signal is assumed to be zero as the temperature should have returned to its nominal state

    ## Arguments

    No additional arguments are currently supported during construction.

    ```python
    import gymnasium as gym
    env = gym.make('Reactor-v3')
    ```


    ## References
    - OSTI report.
    """
    
    # Instantiate the Dymola interface and start Dymola
    dymola = DymolaInterface()
    print(dymola.DymolaVersion())

    #Define Model File
    model = 'ControlTests.SteamTurbine_L2_OpenFeedHeat_Test2'
     
    #Define Previous working directory
    dymola.AddModelicaPath("C:/Users/localuser/Documents/Dymola")

    #open the dymola model in the environment 
    dymola.openModel(model)

    #Add any package dependencies to the enviroment and change working directory
    dymola.openModel("C:/Users/localuser/HYBRID/Models/NHES/package.mo")
    dymola.openModel("C:/Users/localuser/HYBRID/TRANSFORM-Library/TRANSFORM-Library/TRANSFORM/package.mo")
    dymola.ExecuteCommand('Modelica.Utilities.System.setWorkDirectory("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts")') 
    
    #reopen model in new working directory
    dymola.openModel(model)

    #translate the model and create dymosym files
    dymola.translateModel(model)
    
    #Initialise the model and create first dsfinal file
    result = dymola.simulateModel(model, startTime=0, stopTime = 9900, outputInterval=100, method="Esdirk45a",resultFile="SteamTurbine_L2_OpenFeedHeat_Test2")

    #error out if initial simulation fails
    if not result:
        print("Simulation failed. Below is the translation log.")
        log = dymola.getLastErrorLog()
        print(log)
        dymola.exit(1)

    #set time for one step
    dt = 5
    
    #Set a dictionary to hold results based on the Dymola variables
    results = {}
    variables = ["Time","BOP.sensor_pT.T","BOP.sensor_pT.p","BOP.sensor_T2.T","BOP.TCV.dp","pump_SimpleMassFlow1.m_flow", "ramp.y", "boundary.Q_flow_ext","BOP.sensorW.W","FeedForward.y"]
    for key in variables:
        results[key] = []

    #define action space
    AVAIL_MDOT =  [-0.7,-0.2,-0.1, 0 , 0.1, 0.2, 0.7]
    actions_num = 7

    #initialise the class and major variables
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        high = np.array(
            [8, 10, 6, 10], dtype=np.float32
        )
        low = np.array(
            [-8, -10, -6, -10], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(7)
        self.state = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.1, 0.1  # default low
        # )  # default high
        # self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
        #     np.float32
        # )
        
        #define start time
        self.t = 9900
        
        #remake dictionary to hold results
        self.results = {}
        
        for key in self.variables:
            self.results[key] = []
        
        #read in initial results file
        trajsize = self.dymola.readTrajectorySize("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/Original_Temp_Profile.mat")
        signals = self.dymola.readTrajectory("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/Original_Temp_Profile.mat", self.variables, trajsize)
        
        for i in range(0,len(self.variables),1):
            self.results[self.variables[i]].extend(signals[i])
        
        #Set inital state from orginal results
        Tout = self.results["BOP.sensor_pT.T"][-1]
        Pout = self.results["BOP.sensor_pT.p"][-1]
        Tin = self.results["BOP.sensor_T2.T"][-1]
        TCVdp = self.results["BOP.TCV.dp"][-1]
        PumpMFlow = self.results["pump_SimpleMassFlow1.m_flow"][-1]
        Qdemand = self.results["ramp.y"][-1]
        Qout = self.results["BOP.sensorW.W"][-1]
        FF = self.results["FeedForward.y"][-1]
        
        #import in model initial conditions
        self.dymola.ExecuteCommand('importInitial("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/Starting_9900.txt")')
    
        #set initial observation
        yout = [Tout,PumpMFlow, Qout, FF]
        print(yout)
        
        self.state = yout
        
        self.steps_beyond_terminated = None
            
        for i in range(0,len(self.variables),1):
                self.results[self.variables[i]].extend(signals[i])

        return self._get_ob(), {}

    def step(self, a):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        
        #chose feedforward signal based on action number
        FF = self.AVAIL_MDOT[a]


        # Now, augment the state with our pump feedforward action so it can be passed to
        # dymola
        s_augmented = np.append(s, FF)

        #Call main dymola physics dymola returns an observation, results vector and terminal state
        ns, results, terminal = DymolaDyn(self.model, s_augmented, self.t, self.variables, self.results, self.dymola)
        
        print(ns[0])
        self.state = ns
        terminated = terminal
        
        #check within terminal bounds
        if (ns[0]< 664.15 or ns[0]>682.15):
            terminated = True
            
        #increment time
        self.t = self.t + 5
            
        #Define normalise temperature
        x = ns[0]-673.15
        
        #calculate reward based on linear function
        if not terminated:
            reward = 8 - abs(x)
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 8 - abs(x)
        else:
            if self.steps_beyond_terminated == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0
            
        return (self._get_ob(), reward, terminated, False, {})

    #convert state to observation array
    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(
            [s[0]-673.15, s[1]-58, (s[2]-30e6)/1e6, s[3]], dtype=np.float32
        )

    #Unused terminal class  - can be used to calculate a more complex terminal condition
    def _terminal(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return True


#Optional bounding class - again unused
def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)



def DymolaDyn(model ,y0 , t,  variables, results, dymola):
    """
    Dymola dynamics for the time step - sets the feedforward component of the PID controller in the dsin file 
    then simulates 5 seconds of the dynamics
    """
    
    ff = y0[-1]+ y0[-2]
    
    var = "FeedForward.k ="+ str(ff)
    
    print(var)
    dymola.ExecuteCommand(var)

    #calls one time step of the simulation
    result = dymola.simulateModel(model, startTime  = t, stopTime = t+5, numberOfIntervals=0, outputInterval=0.1, method="Esdirk45a", resultFile="SteamTurbine_L2_OpenFeedHeat_Test2")
    
    if result:
        terminal = False

    if not result:
        terminal  = True
        print("Simulation failed. Below is the translation log.")
        log = dymola.getLastErrorLog()
        print(log)
        dymola.exit(1)

    trajsize = dymola.readTrajectorySize("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/SteamTurbine_L2_OpenFeedHeat_Test2.mat")
    signals=dymola.readTrajectory("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/SteamTurbine_L2_OpenFeedHeat_Test2.mat", variables, trajsize)
    
    for i in range(0,len(variables),1):
        results[variables[i]].extend(signals[i])

    Tout = results["BOP.sensor_pT.T"][-1]
    Pout = results["BOP.sensor_pT.p"][-1]
    Tin = results["BOP.sensor_T2.T"][-1]
    TCVdp = results["BOP.TCV.dp"][-1]
    PumpMFlow = results["pump_SimpleMassFlow1.m_flow"][-1]
    Qdemand = results["ramp.y"][-1]
    Qout = results["BOP.sensorW.W"][-1]
    FF = results["FeedForward.y"][-1]
    
    
    yout = [Tout, PumpMFlow, Qout, FF]
    
    #imports the final conditions as initial conditions for the next time step
    dymola.ExecuteCommand('importInitial("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/dsfinal.txt")')
    # We only care about the final timestep and we cleave off action value
    return yout, results, terminal;

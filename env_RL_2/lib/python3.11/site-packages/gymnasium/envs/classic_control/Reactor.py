"""classic Acrobot task"""
from typing import Optional

import numpy as np
from numpy import cos, pi, sin

from dymola.dymola_interface import DymolaInterface
dymola = DymolaInterface()

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

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class ReactorEnv(Env):
    """
    ## Description

    The Acrobot environment is based on Sutton's work in
    ["Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding"](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
    and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
    The system consists of two links connected linearly to form a chain, with one end of
    the chain fixed. The joint between the two links is actuated. The goal is to apply
    torques on the actuated joint to swing the free end of the linear chain above a
    given height while starting from the initial state of hanging downwards.

    As seen in the **Gif**: two blue links connected by two green joints. The joint in
    between the two links is actuated. The goal is to swing the free end of the outer-link
    to reach the target height (black horizontal line above system) by applying torque on
    the actuator.

    ## Action Space

    The action is discrete, deterministic, and represents the torque applied on the actuated
    joint between the two links.

    | Num | Action                                | Unit         |
    |-----|---------------------------------------|--------------|
    | 0   | apply -1 torque to the actuated joint | torque (N m) |
    | 1   | apply 0 torque to the actuated joint  | torque (N m) |
    | 2   | apply 1 torque to the actuated joint  | torque (N m) |

    ## Observation Space

    The observation is a `ndarray` with shape `(6,)` that provides information about the
    two rotational joint angles as well as their angular velocities:

    | Num | Observation                  | Min                 | Max               |
    |-----|------------------------------|---------------------|-------------------|
    | 0   | Cosine of `theta1`           | -1                  | 1                 |
    | 1   | Sine of `theta1`             | -1                  | 1                 |
    | 2   | Cosine of `theta2`           | -1                  | 1                 |
    | 3   | Sine of `theta2`             | -1                  | 1                 |
    | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    where
    - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
    downwards.
    - `theta2` is ***relative to the angle of the first link.***
        An angle of 0 corresponds to having the same angle between the two links.

    The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
    A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.

    ## Rewards

    The goal is to have the free end reach a designated target height in as few steps as possible,
    and as such all steps that do not reach the goal incur a reward of -1.
    Achieving the target height results in termination with a reward of 0. The reward threshold is -100.

    ## Starting State

    Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
    uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.

    ## Episode End

    The episode ends if one of the following occurs:
    1. Termination: The free end reaches the target height, which is constructed as:
    `-cos(theta1) - cos(theta2 + theta1) > 1.0`
    2. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    No additional arguments are currently supported during construction.

    ```python
    import gymnasium as gym
    env = gym.make('Acrobot-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.

    By default, the dynamics of the acrobot follow those described in Sutton and Barto's book
    [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/11/node4.html).
    However, a `book_or_nips` parameter can be modified to change the pendulum dynamics to those described
    in the original [NeurIPS paper](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html).

    ```python
    # To change the dynamics as described above
    env.unwrapped.book_or_nips = 'nips'
    ```

    See the following note for details:

    > The dynamics equations were missing some terms in the NIPS paper which
            are present in the book. R. Sutton confirmed in personal correspondence
            that the experimental results shown in the paper and the book were
            generated with the equations shown in the book.
            However, there is the option to run the domain with the paper equations
            by setting `book_or_nips = 'nips'`


    ## Version History

    - v1: Maximum number of steps increased from 200 to 500. The observation space for v0 provided direct readings of
    `theta1` and `theta2` in radians, having a range of `[-pi, pi]`. The v1 observation space as described here provides the
    sine and cosine of each angle instead.
    - v0: Initial versions release (1.0.0) (removed from gymnasium for v1)

    ## References
    - Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding.
        In D. Touretzky, M. C. Mozer, & M. Hasselmo (Eds.), Advances in Neural Information Processing Systems (Vol. 8).
        MIT Press. https://proceedings.neurips.cc/paper/1995/file/8f1d43620bc6bb580df6e80b0dc05c48-Paper.pdf
    - Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press.
    """

    metadata = {
        "render_modes": [],
        "render_fps": 15,
    }

    dt = 10
    
    results = {}
    variables = ["Time","BOP.sensor_pT.T","BOP.sensor_pT.p","BOP.sensor_T2.T","BOP.TCV.dp","pump_SimpleMassFlow1.m_flow", "ramp.y", "boundary.Q_flow_ext","BOP.sensorW.W","FeedForward.y"]
    for key in variables:
        results[key] = []

    AVAIL_MDOT = [-0.2, -0.1, 0 , 0.1, +0.2]

    torque_noise_max = 0.0

    model = 'ControlTests.SteamTurbine_L2_OpenFeedHeat_Test2'
    
    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 5

    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        high = np.array(
            [800, 200e5, 473, 200e5, 100, 5e7,5e7], dtype=np.float32
        )
        low = np.array(
            [600, 100e5, 373, 1e5, 10, 3e7,3e7], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.state = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        dymola.clear()
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # low, high = utils.maybe_parse_reset_bounds(
        #     options, -0.1, 0.1  # default low
        # )  # default high
        # self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
        #     np.float32
        # )
        
        self.t = 9900
        
        self.results = {}
        
        for key in self.variables:
            self.results[key] = []
            
        dymola.openModel("C:/Users/localuser/HYBRID/Models/NHES/package.mo")
        dymola.openModel("C:/Users/localuser/HYBRID/TRANSFORM-Library/TRANSFORM-Library/TRANSFORM/package.mo")
        dymola.ExecuteCommand('Modelica.Utilities.System.setWorkDirectory("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts")') 
            
        dymola.openModel(self.model)
        
        dymola.translateModel(self.model)
                
        result = dymola.simulateModel(self.model, startTime=0, stopTime = 9900, outputInterval=1, method="Esdirk45a",resultFile="SteamTurbine_L2_OpenFeedHeat_Test2");

        if not result:
            print("Simulation failed. Below is the translation log.")
            log = dymola.getLastErrorLog()
            print(log)
            dymola.exit(1)
            
        trajsize = dymola.readTrajectorySize("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/SteamTurbine_L2_OpenFeedHeat_Test2.mat")
        signals = dymola.readTrajectory("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/SteamTurbine_L2_OpenFeedHeat_Test2.mat", self.variables, trajsize)
        
        for i in range(0,len(self.variables),1):
            self.results[self.variables[i]].extend(signals[i])
        
        Tout = self.results["BOP.sensor_pT.T"][-1]
        Pout = self.results["BOP.sensor_pT.p"][-1]
        Tin = self.results["BOP.sensor_T2.T"][-1]
        TCVdp = self.results["BOP.TCV.dp"][-1]
        PumpMFlow = self.results["pump_SimpleMassFlow1.m_flow"][-1]
        Qdemand = self.results["ramp.y"][-1]
        Qout = self.results["BOP.sensorW.W"][-1]
    
    
        yout = [Tout,Pout, Tin, TCVdp,  PumpMFlow, Qdemand, Qout]
        
        self.state = yout
            
        for i in range(0,len(self.variables),1):
                self.results[self.variables[i]].extend(signals[i])

        return self._get_ob(), {}

    def step(self, a):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        mdot = self.AVAIL_MDOT[a]


        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, mdot)

        #__________________________________________________________________________________EDIT THIS - dymola needs to return a state vector
        ns, results, terminal = DymolaDyn(self.model, s_augmented, self.t, self.variables, self.results)
        #__________________________________________________________________________________EDIT THIS
        
        print(ns)
        self.state = ns
        terminated = terminal
        reward = -1.0 if (s[0]< 672.15 or s[0]>674.15) else 0.0
    
        self.t = self.t + 10
        
        if self.t > 13000:
            terminated = True
            
        return (self._get_ob(), reward, terminated, False, {})

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(
            [s[0], s[1], s[2], s[3], s[4],s[5], s[6]], dtype=np.float32
        )

    def _terminal(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


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

#__________________________________________________________________________________EDIT THIS
#return yout which is a list of 4 observations theta1, theta2, omega1, omega2

def DymolaDyn(model ,y0 , t,  variables, results):
    """
    Dymola dynamics for the time step
    """
    
    dymola.ExecuteCommand('importInitial("C:/Users/localuser/Documents/GitHub/ALPACA/Runscripts/dsfinal.txt")')
    
    ff = y0[-1]
    
    var = "FeedForward.k ="+ str(ff)
    
    print(var)
    dymola.ExecuteCommand(var)

    result = dymola.simulateModel(model, startTime  = t, stopTime = t+10, numberOfIntervals=0, outputInterval=0.1, method="Esdirk45a", resultFile="SteamTurbine_L2_OpenFeedHeat_Test2")
    
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
    
    
    yout = [Tout,Pout, Tin, TCVdp,  PumpMFlow, Qdemand, Qout]
    
    
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout, results, terminal;

#__________________________________________________________________________________EDIT THIS
from typing import SupportsFloat, Any

from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, Box

from polaris.environments import PolarisEnv
import numpy as np


class RepeatedPrisoners(PolarisEnv):
    def __init__(self):

        PolarisEnv.__init__(self, env_id="prisoners")

        self.action_space = Discrete(2)
        self.observation_space = Box(0, 1, (5,), dtype=np.float32)

        self._agent_ids = {0}

        self.t = 0
        self.episode_length = 16

        # we play against a player that collaborates but then has a chance to switch to defecting if defected
        self.opp_action = 0
        self.opp_switch_count = 0
        self.opp_defect_prob = 2e-1

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):

        self.opp_action = 0
        self.opp_switch_count = 0
        self.opp_defect_prob = 2e-1
        self.t = 0
        return {
            0: np.zeros((5,), dtype=np.float32)
        }, {0: {}}

    def step(
        self, action: ActType
    ):
        act = action[0]

        opp_action = int(np.random.random() < self.opp_defect_prob)
        if act == 1 and opp_action == 1:
                self.opp_defect_prob *= 1.5

        if act == opp_action == 0:
            r = 4.
            s = 1
        elif act == 0 and opp_action == 1:
            r = 0.
            s = 2
        elif act == 1 and opp_action == 0:
            r = 5.
            s = 3
        else:
            r = 1.
            s = 4



        self.t += 1

        done = self.t == self.episode_length

        state = np.zeros((5,), dtype=np.float32)
        state[s] = 1.
        #state[-1] = self.t /  self.episode_length


        return {0: state}, {0: r*0.2}, {0: done, "__all__": done}, {0: False, "__all__": False}, {0: {}}





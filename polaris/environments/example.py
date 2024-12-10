from typing import Optional

import numpy as np
import tree
from gymnasium.envs.classic_control import CartPoleEnv
from gymnasium.envs.box2d import LunarLander

from polaris.policies.random import RandomPolicy
from .polaris_env import PolarisEnv
from ray.tune.registry import ENV_CREATOR, _global_registry
from polaris.experience.episode import Episode
from polaris.experience.sampling import SampleBatch

from gymnasium.spaces import Dict, Discrete, Box

class DummyEnv(PolarisEnv):

    def __init__(self):
        PolarisEnv.__init__(self, env_id="dummy")
        self._agent_ids = {0,1,2}
        self.num_players = len(self._agent_ids)
        self.episode_length = 32
        self.step_count = 0

        self.action_space = Discrete(3)
        self.observation_space = Dict({
            "continuous": Box(0, 1, shape=(3,)),
            "discrete": Discrete(5),
        })

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.step_count = 0
        return {aid: self.observation_space.sample() for aid in self.get_agent_ids()}, {aid: {} for aid in self.get_agent_ids()}

    def step(
            self,
            action
    ):
        self.step_count += 1
        trunc = {
            "__all__": self.step_count == self.episode_length
        }
        dones = {
            aid: False for aid in self.get_agent_ids()
        }
        dones["__all__"] = False
        rews = {
            aid: np.random.normal() for aid in self.get_agent_ids()
        }
        obs = {
            aid: self.observation_space.sample() for aid in self.get_agent_ids()
        }
        info = {
            aid: {} for aid in self.get_agent_ids()
        }
        return obs, rews, dones, trunc, info


class PolarisCartPole(CartPoleEnv, PolarisEnv):
    env_id = "PolarisCartPole"

    def __init__(self, env_index, **config):

        PolarisEnv.__init__(self, env_index=env_index)
        CartPoleEnv.__init__(self)

        self._agent_ids = {0}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        return tree.map_structure(lambda v: {
            0: v
        }, super().reset(seed=seed, options=options))


    def step(self, action):
        d = tree.map_structure(lambda v: {
            0: v
        }, super().step(action[0]))
        d[2]["__all__"] = d[2][0]
        d[3]["__all__"] = False
        return d


class PolarisLunarLander(LunarLander, PolarisEnv):

    env_id = "lunarlander"

    def __init__(self, *args, env_index=None, **config):
        PolarisEnv.__init__(self, env_index=env_index, **config)
        LunarLander.__init__(self, *args, **config)

        self._agent_ids = {0}
        self.t = 0
        self.max_t = 8192*2

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # makes a call to step
        self.t = 0
        return super().reset(seed=seed, options=options)



    def step(self, action):
        if isinstance(action, int):
            action = {0: action}

        d = tree.map_structure(lambda v: {
            0: v
        }, super().step(action[0]))

        self.t += 1
        timeout = self.t >= self.max_t
        d[2][0] = d[2][0] or timeout
        d[2]["__all__"] = d[2][0]
        d[3]["__all__"] = False
        d[1][0] = d[1][0] #* 0.01


        return d


from typing import SupportsFloat, Any

import ray.tune
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
from ray.tune.registry import _global_registry, ENV_CREATOR

class PolarisEnv(Env):

    def __init__(self, env_id, **config):

        self.env_id = env_id
        self.action_space: Space
        self.observation_space: Space
        self.num_players: int
        #self.episode_length: int

        super().__init__()
        self._agent_ids = set()
        self.config = config

    def register(self):
        ray.tune.register_env(self.env_id, lambda: self.__class__(**self.config))

    def get_agent_ids(self):
        return self._agent_ids

    @staticmethod
    def make(env_id, **config):
        return _global_registry.get(ENV_CREATOR, env_id)(**config)


from typing import SupportsFloat, Any

import ray.tune
from gymnasium import Env
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
from ray.tune.registry import _global_registry, ENV_CREATOR


class PolarisEnv(Env):

    env_id = "abstract"

    @classmethod
    def register(cls):

        def env_maker(env_index=None, **config):
            return cls(env_index=env_index, **config)

        ray.tune.register_env(cls.env_id, env_maker)

    @staticmethod
    def make(env_id, env_index=None, **config):
        return _global_registry.get(ENV_CREATOR, env_id)(env_index=env_index, **config)

    def __init__(self, env_index=None, **config):

        self.action_space: Space
        self.observation_space: Space
        self.num_players: int
        #self.episode_length: int

        super().__init__()
        self._agent_ids = set()
        self.config = config

        self.env_index = env_index



    def get_agent_ids(self):
        return self._agent_ids


    def get_episode_metrics(self):
        return {}

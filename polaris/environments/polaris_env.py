import ray.tune
from gymnasium import Env
from gymnasium.spaces import Space

class PolarisEnv(Env):

    def __init__(self, env_id, **config):
        self.env_id = env_id
        self.action_space: Space
        self.observation_space = Space
        super().__init__()
        self._agent_ids = set()
        self.config = config

    def register(self):
        ray.tune.register_env(self.env_id, lambda config: self.__class__(env_id=self.env_id, **self.config))

    def get_agent_ids(self):
        return self._agent_ids



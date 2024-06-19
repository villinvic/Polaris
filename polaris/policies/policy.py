from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, Type

from gymnasium.spaces import Space
from ml_collections import ConfigDict
from polaris.models.base import BaseModel

from polaris import SampleBatch


class Policy(ABC):
    policy_type = "abstract"

    def __init__(
            self,
            name: str,
            action_space: Space,
            observation_space: Space,
            model: Type[BaseModel],
            config: ConfigDict,
    ):
        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space
        self.version = 0
        self.config = config
        self.policy_config = config.policy_config
        self.model_class = model

        self.init_model()

    def init_model(self):
        self.model = self.model_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config=self.config.model_config
        )

    @abstractmethod
    def compute_action(
            self,
            input_dict: SampleBatch
            # observation,
            # state=None,
            # prev_action=None,
            # prev_obs=None,
            # prev_reward=None,
    ):
        pass

    def get_initial_state(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, version={self.version})"

    def setup(self, policy_params: "PolicyParams"):
        if policy_params.policy_type != self.policy_type:
            raise ValueError(f"Policy type is not the same: {self.get_params()} != {policy_params}")

        self.name = policy_params.name
        self.set_weights(weights=policy_params.weights)
        self.version = policy_params.version

        return self

    @abstractmethod

    def get_weights(self):
        return {}

    @abstractmethod
    def set_weights(self, weights: Dict):
        pass

    def get_params(self):
        return PolicyParams(
            name=self.name,
            weights=self.get_weights(),
            config=self.policy_config,
            version=self.version,
            policy_type=self.policy_type
        )

    def train(self, batch: SampleBatch):
        self.version += 1


class DummyPolicy(Policy):
    policy_type = "parametrised"

    def compute_action(
            self,
            observation,
            states=None,
            prev_action=None,
            prev_reward=None,
    ):
        return 0, None


class PolicyParams(NamedTuple):
    """Describes the specs of a policy"""

    name: str = "unnamed"
    weights: dict = {}
    config: ConfigDict = ConfigDict()
    version: int = 0
    policy_type: str = "parametrised"

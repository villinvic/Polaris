import importlib
import time
from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, Type

import numpy as np
from gymnasium.spaces import Space
from ml_collections import ConfigDict

from polaris.experience import SampleBatch

class Policy(ABC):
    policy_type = "abstract"

    def __init__(
            self,
            name: str,
            action_space: Space,
            observation_space: Space,
            config: ConfigDict,
            options: ConfigDict,
            policy_config: ConfigDict,
            stats = None,
            **kwargs,
    ):
        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space
        self.version = 1
        self.options = options
        self.stats = {} if stats is None else stats
        self.config = config
        self.policy_config = policy_config
        self.model_class = getattr(importlib.import_module(self.config.model_path), self.config.model_class)
        self.is_recurrent = self.model_class is not None and self.model_class.is_recurrent
        self.model = None
        self.init_model()

    def init_model(self):
        self.model = self.model_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.policy_config
        )
        self.model.setup()

    @abstractmethod
    def compute_action(
            self,
            input_dict: SampleBatch
    ):
        pass

    def get_initial_state(self):
        return self.model.get_initial_state()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, version={self.version})"

    def setup(self, policy_params: "PolicyParams"):
        if policy_params.policy_type != self.policy_type:
            raise ValueError(f"Policy type is not the same: {self.get_params()} != {policy_params}")

        self.name = policy_params.name
        self.set_weights(weights=policy_params.weights)
        self.policy_config.update(**policy_params.config)
        self.version = policy_params.version
        self.options = policy_params.options
        self.stats = policy_params.stats
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
            options=self.options,
            version=self.version,
            stats=self.stats,
            policy_type=self.policy_type
        )

    def train(self, batch: SampleBatch):
        res =  {"delta_version": self.version - np.mean(batch[SampleBatch.VERSION][batch[SampleBatch.VERSION]>0])}
        self.version += 1
        return res


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
    options: ConfigDict = ConfigDict()
    stats: dict = {}
    version: int = 0
    policy_type: str = "parametrised"

class ParamsMap(dict):
    pass

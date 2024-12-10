import importlib
import time
from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, Type, Union, Any, Tuple

import numpy as np
from gymnasium.spaces import Space
from ml_collections import ConfigDict
from polaris.models import BaseModel

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
            stats: Union[None, dict] = None,
            **kwargs,
    ):
        """
        Base policy class.

        :param name: Name of the policy.
        :param action_space: Action space of the policy.
        :param observation_space: Observation space of the policy.
        :param config: Config of the trainer.
        :param options: Additional options for the policy (usually for the environment, such as a policy type).
        :param policy_config: Config for the policy (two policies can have different configs).
        :param stats: Stat records of the policy.
        """

        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space
        self.version = 1
        self.options = options
        self.stats = {"samples_generated": 0} if stats is None else stats
        self.config = config
        self.policy_config = policy_config
        self.model_class = getattr(importlib.import_module(self.config.model_path), self.config.model_class)
        self.is_recurrent = self.model_class is not None and self.model_class.is_recurrent
        self.model: Union[None, BaseModel] = None
        self.init_model()

    def init_model(self):
        """
        Method called at policy instantiation to initialise the model.
        """
        self.model = self.model_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            config=self.policy_config
        )
        self.model.setup()

    @abstractmethod
    def compute_single_action(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
    ) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def compute_single_action_with_extras(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
    ) -> Tuple[Any, Any, dict]:
        pass

    @abstractmethod
    def compute_value_batch(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens,
    ) -> Any:
        pass

    def get_initial_state(self):
        """
        Returns the state of a policy at episode initialisation.
        Could be an RNN state, for example.
        """
        return self.model.get_initial_state()

    def setup(
            self,
            policy_params: "PolicyParams"
    ) -> "Policy":
        """
        Sets the policy with the specs given in the provided params.
        """

        if policy_params.policy_type != self.policy_type:
            raise ValueError(f"Policy type is not the same: {self.get_params()} != {policy_params}")

        self.name = policy_params.name
        self.set_weights(weights=policy_params.weights)
        self.policy_config.update(**policy_params.config)
        self.version = policy_params.version
        self.options = policy_params.options
        self.stats = policy_params.stats
        return self

    def get_weights(self) -> dict:
        return {}

    @abstractmethod
    def set_weights(
            self,
            weights: Dict
    ):
        pass

    def get_params(self) -> "PolicyParams":
        """
        Converts the policy into a PolicyParam object.
        """
        return PolicyParams(
            name=self.name,
            weights=self.get_weights(),
            config=self.policy_config,
            options=self.options,
            version=self.version,
            stats=self.stats,
            policy_type=self.policy_type
        )

    def train(
            self,
            batch: SampleBatch
    ):
        """
        Subclasses should call this super method to update the version of the policy.
        """

        res =  {"delta_version": self.version - np.mean(batch[SampleBatch.VERSION][batch[SampleBatch.VERSION]>0])}
        self.version += 1
        return res

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, version={self.version})"


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

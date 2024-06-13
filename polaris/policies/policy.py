from abc import ABC, abstractmethod
from typing import NamedTuple

from gymnasium.spaces import Space

class Policy(ABC):

    def __init__(
            self,
            name: str,
            action_space: Space,
            observation_space: Space
    ):
        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space

        self.version = 0

    @abstractmethod
    def compute_action(
            self,
            observation,
            states=None,
            prev_action=None,
            prev_reward=None

    ):
        pass

    def get_initial_state(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, version={self.version})"

    def setup(self, policy_params: "PolicyParams"):
        return self


class DummyPolicy(Policy):

    def compute_action(
            self,
            observation,
            states=None,
            prev_action=None,
            prev_reward=None

    ):
        return 0


class PolicyParams(NamedTuple):
    """Describes the specs of a policy"""

    name: str = "unnamed"
    weights: dict = {}
    version: int = 0
    policy_type: str = "parametrised"

from abc import ABC, abstractmethod
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

    @abstractmethod
    def compute_action(
            self,
            observation,
            state=None,
            prev_action=None,
            prev_reward=None

    ):
        pass

    def get_initial_state(self):
        pass

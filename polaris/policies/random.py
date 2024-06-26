from typing import Dict

from .policy import Policy

class RandomPolicy(Policy):
    policy_type = "random"

    def __init__(
            self,
            action_space,
            config
    ):
        super().__init__(
            name="RandomPolicy",
            action_space=action_space,
            observation_space=None,
            config=config,
            policy_config=None
        )

    def compute_action(
            self,
            input_dict

    ):
        return self.action_space.sample(), None

    def get_weights(self):
        return {}

    def set_weights(self, weights: Dict):
        pass

    def init_model(self):
        pass
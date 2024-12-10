from typing import Dict, Tuple, Any

import numpy as np
from polaris.experience import SampleBatch

from .policy import Policy

class RandomPolicy(Policy):
    policy_type = "random"

    def __init__(
            self,
            action_space,
            config
    ):
        """
        Random policy. Picks actions at uniform.
        """

        super().__init__(
            name="RandomPolicy",
            action_space=action_space,
            observation_space=None,
            config=config,
            options={},
            policy_config=None
        )

    def compute_single_action(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
    ) -> Tuple[Any, Any]:
        return self.action_space.sample(), None

    def compute_single_action_with_extras(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
    ) -> Tuple[Any, Any, dict]:
        act, state = self.compute_single_action(
            obs=obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
            state=state
        )
        return act, state, {SampleBatch.VALUES: 0.}

    def compute_value_batch(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens,
    ) -> Any:
        return np.zeros_like(prev_reward)

    def get_weights(self):
        return {}

    def set_weights(self, weights: Dict):
        pass

    def init_model(self):
        pass
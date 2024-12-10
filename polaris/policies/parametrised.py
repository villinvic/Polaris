from typing import Dict, Any, Tuple

import numpy as np

from .policy import Policy
import tree
import tensorflow as tf

from polaris.experience import SampleBatch

class ParametrisedPolicy(Policy):
    """
    A parametrised policy has a tensorflow model as its core.
    """

    policy_type = "parametrised"

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {v.name: v.numpy()
                for v in self.model.trainable_variables}

    def set_weights(self, weights: Dict):
        x = {v.name: v for v in self.model.trainable_variables}
        print(list(weights.keys()))
        for name, w in weights.items():
            x[name].assign(w)

    def compute_single_action(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
    ) -> Tuple[Any, Any]:
        return self.model.compute_single_action(
            obs,
            prev_action,
            prev_reward,
            state,
        )

    def compute_single_action_with_extras(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
    ) -> Tuple[Any, Any, dict]:
        return self.model.compute_single_action_with_extras(
            obs,
            prev_action,
            prev_reward,
            state,
        )

    def compute_value_batch(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens,
    ) -> Any:
        return self.model(
            obs=obs,
            prev_action=prev_action,
            prev_reward=prev_reward,
            state=state,
            seq_lens=seq_lens,
        )[2][SampleBatch.VALUES]
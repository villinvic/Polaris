from typing import Dict

import numpy as np

from .policy import Policy
import tree
import tensorflow as tf

from polaris.experience import SampleBatch

class ParametrisedPolicy(Policy):
    policy_type = "parametrised"

    def compute_action(
            self,
            input_dict
    ):
        return self.model.compute_action(input_dict)

    def compute_value(self, input_dict):
        return self.model.compute_value(input_dict)

    @tf.function
    def _compute_action_dist(
            self,
            input_dict
    ):
        return self.model(input_dict)



    def get_weights(self) -> Dict[str, np.ndarray]:
        return {v.name: v.numpy()
                for v in self.model.trainable_variables}

    def set_weights(self, weights: Dict):
        x = {v.name: v for v in self.model.trainable_variables}
        for name, w in weights.items():
            x[name].assign(w)
from typing import Tuple, Any

import sonnet as snt
import numpy as np
from gymnasium.spaces import Discrete
import tensorflow as tf
from .base import BaseModel
from .utils.categorical_distribution import CategoricalDistribution
from polaris.experience import SampleBatch


class FCModel(BaseModel):
    """
    TODO: Out-of-date
    We expect users to code their own model.
    This one expects a box as observation and a discrete space for actions
    """

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(FCModel, self).__init__(
            name="FCModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.output_size = action_space.n
        self.optimiser = snt.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=1e-5,
        )

        self.action_dist = CategoricalDistribution

        self._mlp = snt.nets.MLP(
            output_sizes=self.config.fc_dims,
            activate_final=True,
            name="MLP"
        )
        self._pi_out = snt.Linear(
            output_size=self.output_size,
            name="action_logits"
        )
        self._value_out = snt.Linear(
            output_size=1,
            name="values"
        )

    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        return self._mlp(tf.expand_dims(obs, axis=0))

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        return self._mlp(obs)

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        final_embeddings = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        policy_logits = self._pi_out(final_embeddings)
        extras = {
            SampleBatch.VALUES: tf.squeeze(self._value_out(final_embeddings))
        }
        return policy_logits, state, extras

    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        final_embeddings = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        return self._pi_out(final_embeddings), state

    def __call__(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ) -> Tuple[Any, Any]:
        final_embeddings = self.batch_input(
            obs,
            prev_action,
            prev_reward,
            state
        )
        policy_logits = self._pi_out(final_embeddings)
        self._values = tf.squeeze(self._value_out(final_embeddings))
        return policy_logits, self._values

    def critic_loss(
            self,
            vf_targets
    ):
        return tf.math.square(vf_targets - self._values)

    def get_initial_state(self):
        return (np.zeros(1, dtype=np.float32),)

    def get_metrics(self):
        return {}


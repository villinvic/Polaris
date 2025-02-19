from typing import Tuple, Any

import numpy as np
from gymnasium import Space
from ml_collections import ConfigDict
from polaris.experience import SampleBatch
from polaris.models import BaseModel
from polaris.models.utils import CategoricalDistribution


import tensorflow as tf
import sonnet as snt


class CartPoleModel(BaseModel):
    """
    To debug
    """
    is_recurrent = False

    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            config: ConfigDict,
    ):
        super().__init__(
            name="CartPoleModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )

        self.action_dist = CategoricalDistribution
        self.optimiser = snt.optimizers.Adam(learning_rate=config.lr, epsilon=1e-5)

        self.final_mlp = snt.nets.MLP([64, 64], activate_final=True)
        self.policy_head = snt.Linear(self.action_space.n, name="policy_head")
        self._value_logits = None
        self.value_head = snt.Linear(1, name="value_head")


    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):

        return self.final_mlp(obs)

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        return self.final_mlp(obs)


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

        policy_logits = self.policy_head(final_embeddings)
        extras = {
            SampleBatch.VALUES: tf.squeeze(self.value_head(final_embeddings))
        }
        return policy_logits, state, extras

    def compute_action_batch_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        """
        Computes action, with next_states and extras.
        Should at least have action_logp, action_logits, values, kwargs in the extras.
        """
        actions, action_logits, action_logps, values = self._compute_action_batch_with_extras(
            obs,
            prev_action,
            prev_reward,
            state
        )

        return (
            actions.numpy(),
            state,
            action_logits.numpy(),
            action_logps.numpy(),
            values.numpy()
        )

    @tf.function(jit_compile=False)
    def _compute_action_batch_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
    ):
        action_logits, state, values = self.forward_action_batch_with_extras(
            obs,
            prev_action,
            prev_reward,
            state
        )

        action_dist = self.action_dist(action_logits)
        actions = action_dist.sample()
        logps = action_dist.logp(actions)

        return actions, action_logits, logps, values

    def forward_action_batch_with_extras(
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
        self._values = tf.squeeze(self.value_head(final_embeddings))
        return self.policy_head(final_embeddings), state, self._values

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
        return self.policy_head(final_embeddings), state

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
        policy_logits = self.policy_head(final_embeddings)
        self._values = tf.squeeze(self.value_head(final_embeddings))
        return policy_logits, self._values

    def critic_loss(
            self,
            vf_targets
    ):
        return tf.math.square(vf_targets-self._values)

    def get_initial_state(self):
        return (np.zeros(2, dtype=np.float32),)

    def get_metrics(self):
        return {}


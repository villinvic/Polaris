from abc import abstractmethod
from typing import Union, Tuple, Any

import numpy as np
import sonnet as snt
import tree
from gymnasium.spaces import Space, Dict, Box, Discrete
from ml_collections import ConfigDict
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import time

from polaris.experience.episode import batchify_input
from polaris.experience import SampleBatch


class BaseModel(snt.Module):
    is_recurrent = False

    def __init__(
            self,
            name: str,
            observation_space: Space,
            action_space: Space,
            config: ConfigDict,
    ):
        super(BaseModel, self).__init__(name=name)

        self.action_space = action_space
        self.observation_space = observation_space
        self.num_outputs = None
        self.config = config
        self.optimiser: Union[None, Optimizer] = None

        self._values = None
        self.action_dist = None

    def compute_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
    ) -> Tuple[Any, Any]:
        """
        This is supposed to be only used when computing actions in spectator workers.
        """
        action, state = self._compute_single_action(
            obs,
            prev_action,
            prev_reward,
            state,
        )
        return action.numpy(), state

    @tf.function(jit_compile=False)
    def _compute_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
    ):
        action_logits, state = self.forward_single_action(
            obs,
            prev_action,
            prev_reward,
            state
        )
        action_logits = tf.squeeze(action_logits)
        action_dist = self.action_dist(action_logits)
        action = action_dist.sample()
        return action, state

    def compute_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ) -> Tuple[Any, Any, dict]:
        """
        Computes action, with next_states and extras.
        Should at least have action_logp, action_logits, values, kwargs in the extras.
        """
        t = time.time()
        action, state, extras = self._compute_single_action_with_extras(
            obs,
            prev_action,
            prev_reward,
            state
        )
        extras = tree.map_structure(lambda v: v.numpy(), extras)
        extras["compute_single_action_with_extras_ms"] = time.time() - t
        return action.numpy(), tree.map_structure(lambda v: v.numpy(), state), extras

    @tf.function(jit_compile=False)
    def _compute_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
    ):
        action_logits, state, extras = self.forward_single_action_with_extras(
            obs,
            prev_action,
            prev_reward,
            state
        )

        action_logits = tf.squeeze(action_logits)
        action_dist = self.action_dist(action_logits)
        action = action_dist.sample()
        logp = action_dist.logp(action)

        action_logits = tf.squeeze(action_logits)
        action_dist = self.action_dist(action_logits)
        action = action_dist.sample()

        extras[SampleBatch.ACTION_LOGP] = logp
        extras[SampleBatch.ACTION_LOGITS] = action_logits

        return action, state, extras

    def setup(self):
        """
        Initialise the layers and model related variables.
        """
        # TODO: is this fine to pick an arbitrary shape of batch ?
        T = 5
        B = 3

        x = tree.map_structure(
            lambda v: 1. if not isinstance(v, np.ndarray) else np.ones_like(v),
            self.observation_space.sample()
        )
        dummy_obs = tree.map_structure(
            lambda v: np.ones_like(v, shape=(T, B) + v.shape),
            x
        )
        dummy_reward = np.ones((T, B), dtype=np.float32)
        dummy_action = np.zeros((T, B), dtype=np.int32)

        dummy_state_0 = self.get_initial_state()

        dummy_state = tree.map_structure(
            lambda v: np.repeat(v, B, axis=0), dummy_state_0
        )
        seq_lens = np.ones((B,), dtype=np.int32) * T

        batch_logits, batch_values = self(
            obs=dummy_obs,
            prev_action=dummy_action,
            prev_reward=dummy_reward,
            state=dummy_state,
            seq_lens=seq_lens
        )

        inputs = batchify_input(
            obs=x,
            prev_action=dummy_action[0, :1],
            prev_reward=dummy_reward[0, :1],
            state=dummy_state_0,
        )

        _, _ = self.compute_single_action(
            **inputs
        )
        _, _, extras = self.compute_single_action_with_extras(
            **inputs
        )
        logits = extras[SampleBatch.ACTION_LOGITS]
        value = extras[SampleBatch.VALUES]

        batch_logits = batch_logits.numpy()
        batch_values = batch_values.numpy()

        assert np.isclose(batch_values[0, 0], value, atol=1e-3), \
            f"Batch and sample computation do not match on values: {(batch_values[0, 0], value)}"
        assert np.allclose(batch_logits[0, 0], logits, atol=1e-3), \
            f"Batch and sample computation do not match on logits: {(batch_logits[0, 0], logits)}"


    def get_initial_state(self):
        """
        By default, the policy does not have a state.
        """
        return None

    # tf methods
    @abstractmethod
    def __call__(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens,
    ) -> Tuple[Any, Any]:
        """
        Used by the policy when learning.
        The inputs are in batch [T, B, ...],
        except seq_lens that is of shape [B].
        Should return action logits and values.
        """
        raise NotImplementedError("To be implemented by subclasses.")

    @abstractmethod
    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        raise NotImplementedError("To be implemented by subclasses.")

    @abstractmethod
    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        raise NotImplementedError("To be implemented by subclasses.")


    @abstractmethod
    def critic_loss(
            self,
            vf_targets
    ):
        raise NotImplementedError("To be implemented by subclasses.")

    def get_metrics(self) -> dict:
        return {}




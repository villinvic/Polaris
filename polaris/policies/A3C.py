from typing import Dict, List

import tree
from gymnasium import Space
from ml_collections import ConfigDict

from .policy import Policy
from polaris import SampleBatch
from polaris.models.utils.gae import compute_gae
import tensorflow as tf
from polaris.models.base import BaseModel
from polaris.models.utils.misc import explained_variance
import numpy as np


class A3CPolicy(Policy):
    policy_type = "parametrised"

    def __init__(
            self,
            name: str,
            action_space: Space,
            observation_space: Space,
            model: BaseModel,
            config: ConfigDict,
    ):
        super().__init__(
            name=name,
            action_space=action_space,
            observation_space=observation_space,
            model=model,
            config=config
        )

    def init_model(self):
        super().init_model()
        x = self.observation_space.sample()
        dummy_obs = np.zeros_like(x, shape=(1,) + x.shape)
        dummy_reward = np.zeros((1,), dtype=np.float32)
        dummy_actions = np.zeros_like((1,), dtype=np.int32)

        self.model({
                SampleBatch.OBS: dummy_obs,
                SampleBatch.REWARD: dummy_reward,
                SampleBatch.ACTION: dummy_actions,
             }
        )

    def train(
            self,
            input_batch: SampleBatch
    ):
        input_batch[SampleBatch.OBS] = np.concatenate([input_batch[SampleBatch.OBS], input_batch[SampleBatch.OBS][:, -1:]], axis=1)
        super().train(input_batch)

        metrics = self._train(
            input_batch=input_batch
        )

        return metrics


    @tf.function
    def _train(
            self,
            input_batch: SampleBatch
    ):
        B, T = tf.shape(input_batch[SampleBatch.OBS])
        with tf.GradientTape() as tape:
            with tf.device('/gpu:0'):
                action_logits, state = self.model(
                    input_batch
                )
                action_logits = action_logits[:-1]
                action_logp = self.model.action_dist.logp(action_logits)
                entropy = self.model.action_dist.entropy()
                vf_preds = self.model.value_function()

                smoothed_returns = compute_gae(
                rewards=input_batch[SampleBatch.REWARD],
                dones=input_batch[SampleBatch.DONE],
                values=vf_preds[:, -1],
                discount=self.policy_config.discount,
                gae_lambda=self.policy_config.gae_lambda,
                bootstrap_v= vf_preds[:, -1]
                )

                advantage = tf.square(vf_preds - smoothed_returns)

                critic_loss = 0.5 * advantage
                policy_loss = action_logp * tf.stop_gradient(advantage) - self.policy_config.entropy_cost * entropy

                total_loss = tf.reduce_mean(policy_loss + critic_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimiser.apply_gradient(gradients)

        mean_entropy = tf.reduce_mean(entropy)
        mean_vf_loss = tf.reduce_mean(critic_loss)
        mean_pi_loss = tf.reduce_mean(policy_loss)
        mean_grad_norm = tf.reduce_mean(gradients)
        max_grad_norm = tf.reduce_max(gradients)
        explained_vf = explained_variance(
            tf.reshape(smoothed_returns, [-1]),
            tf.reshape(vf_preds, [-1])
        )

        return {
            "mean_entropy": mean_entropy,
            "mean_vf_loss": mean_vf_loss,
            "mean_pi_loss": mean_pi_loss,
            "mean_grad_norm": mean_grad_norm,
            "max_grad_norm": max_grad_norm,
            "explained_vf": explained_vf,
        }

    def compute_action(
            self,
            input_dict
    ):

        action_logits, state = self._compute_action_dist(
            tree.map_structure(lambda v: [v], input_dict)
        )
        action_logits = tf.squeeze(action_logits).numpy()
        action_dist = self.model.action_dist(action_logits)
        action = action_dist.sample()
        logp = action_dist.logp(action).numpy()
        return action.numpy(), state, logp, action_logits

    @tf.function-
    def _compute_action_dist(
            self,
            input_dict
    ):
        return self.model(input_dict)


    def get_weights(self):
        return {v.name: v.numpy()
                for v in self.model.trainable_variables}

    def set_weights(self, weights: List):
        x = {v.name: v for v in self.model.trainable_variables}
        for name, v in x.items():
            v.assign(weights[name])







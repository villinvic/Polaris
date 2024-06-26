from typing import Dict, List, Type

import tree
from gymnasium import Space
from ml_collections import ConfigDict

from .parametrised import ParametrisedPolicy
from polaris import SampleBatch
from polaris.models.utils.gae import compute_gae
import tensorflow as tf
from polaris.models.base import BaseModel
from polaris.models.utils.misc import explained_variance
import numpy as np


class A3CPolicy(ParametrisedPolicy):
    policy_type = "parametrised"

    def __init__(
            self,
            name: str,
            action_space: Space,
            observation_space: Space,
            model: Type[BaseModel],
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
            input_batch=dict(input_batch)
        )

        return metrics

    @tf.function
    def _train(
            self,
            input_batch
    ):
        #B, T = tf.shape(input_batch[SampleBatch.OBS])
        with tf.GradientTape() as tape:
            with tf.device('/gpu:0'):
                action_logits, state = self.model(
                    input_batch
                )
                action_logits = action_logits[:, :-1]
                action_dist = self.model.action_dist(action_logits)
                action_logp = action_dist.logp(input_batch[SampleBatch.ACTION])
                entropy = action_dist.entropy()
                all_vf_preds = self.model.value_function()
                vf_preds = all_vf_preds[:, :-1]
                next_vf_pred = all_vf_preds[:, -1]

                values = compute_gae(
                rewards=input_batch[SampleBatch.REWARD],
                dones=input_batch[SampleBatch.DONE],
                values=vf_preds,
                discount=self.policy_config.discount,
                gae_lambda=self.policy_config.gae_lambda,
                bootstrap_v=next_vf_pred
                )

                #values = input_batch[SampleBatch.REWARD] + next_vf_pred * self.policy_config.discount * (1. - input_batch[SampleBatch.DONE])
                advantage = values - vf_preds
                critic_loss =  0.5 * tf.square(advantage)
                policy_loss = - action_logp * tf.stop_gradient(advantage) - self.policy_config.entropy_cost * entropy

                total_loss = tf.reduce_mean(policy_loss + critic_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimiser.apply_gradients(zip(gradients,self.model.trainable_variables))

        mean_entropy = tf.reduce_mean(entropy)
        mean_vf_loss = tf.reduce_mean(critic_loss)
        mean_pi_loss = tf.reduce_mean(policy_loss)
        total_grad_norm = 0.
        num_params = 0
        for grad in gradients:
            total_grad_norm += tf.reduce_sum(tf.abs(grad))
            num_params += tf.size(grad)
        mean_grad_norm = total_grad_norm / tf.cast(num_params, tf.float32)
        explained_vf = explained_variance(
            tf.reshape(values, [-1]),
            tf.reshape(vf_preds, [-1])
        )

        return {
            "mean_entropy": mean_entropy,
            "mean_vf_loss": mean_vf_loss,
            "mean_pi_loss": mean_pi_loss,
            "mean_grad_norm": mean_grad_norm,
            "explained_vf": explained_vf,
        }








import time
from typing import Dict

from polaris.policies.utils.batch_processing import make_time_major
from .parametrised import ParametrisedPolicy
from polaris.experience import SampleBatch, get_epochs

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from polaris.policies.utils.misc import explained_variance


class PPO(ParametrisedPolicy):

    needed_keys = [SampleBatch.OBS,
            SampleBatch.OBS,
            SampleBatch.ACTION,
            SampleBatch.PREV_ACTION,
            SampleBatch.PREV_REWARD,
            SampleBatch.STATE,
            SampleBatch.SEQ_LENS,
            SampleBatch.ACTION_LOGITS,
            SampleBatch.ACTION_LOGP,
            SampleBatch.ADVANTAGES,
            SampleBatch.VF_TARGETS
            ]

    def __init__(
            self,
            *args,
            config=None,
            policy_config=None,
            options=None,
            **kwargs
    ):

        self.kl_coeff = tf.Variable(
            policy_config.initial_kl_coeff,
            trainable=False,
            dtype=tf.float32)

        super().__init__(
            *args,
            config=config,
            policy_config=policy_config,
            options=options,
            **kwargs
        )

    def get_weights(self) -> Dict[str, np.ndarray]:
        w = super().get_weights()
        w["kl_coeff"] = self.kl_coeff.numpy()
        return w

    def set_weights(self, weights: Dict):
        kl_coeff = weights.pop("kl_coeff", None)
        if kl_coeff is not None:
            self.kl_coeff.assign(kl_coeff)
        else:
            print("No kl_coeff ???")

        super().set_weights(weights)

    def train(
            self,
            input_batch: SampleBatch
    ):
        preprocess_start_time = time.time()
        res = super().train(input_batch)

        to_del = []

        for k in input_batch:
            if k not in self.needed_keys:
                to_del.append(k)
        for k in to_del:
            del input_batch[k]

        tm_input_batch = make_time_major(input_batch)

        nn_train_time = time.time()

        # normalise advantages

        adv = tm_input_batch[SampleBatch.ADVANTAGES]
        tm_input_batch[SampleBatch.ADVANTAGES][:] = (adv-np.mean(adv)) / np.maximum(1e-4, np.std(adv))

        for minibatch in get_epochs(tm_input_batch,
                                    n_epochs=self.config.n_epochs,
                                    minibatch_size=self.config.minibatch_size
                                    ):

            metrics = self._train(
                **minibatch
            )
            di = {
                m: v.numpy() for m,v  in metrics.items()
            }
            del metrics["x"]
            del metrics["y"]

            print(di["x"], di["y"])


        last_kl = metrics["kl"]
        kl_coeff_val = self.kl_coeff.value()
        if kl_coeff_val > 0.:
            # Increase.
            if last_kl > 2.0 * self.policy_config.kl_target:
                kl_coeff_val *= 1.5
                self.kl_coeff.assign(kl_coeff_val)
                # Decrease.
            elif last_kl < 0.5 * self.policy_config.kl_target:
                kl_coeff_val *= 0.5
                self.kl_coeff.assign(kl_coeff_val)
        metrics.update(**res,
                       preprocess_time_ms=(nn_train_time-preprocess_start_time)*1000.,
                       grad_time_ms=(time.time() - nn_train_time) * 1000.)

        return metrics


    @tf.function
    def _train(
            self,
            *,
            obs,
            action,
            prev_action,
            prev_reward,
            state,
            seq_lens,
            action_logits,
            action_logp,
            advantages,
            vf_targets,
    ):
        """
        If an auxiliary loss is required,
        subclass the PPO class. The parameters of the _train function may not be enough.
        """

        with tf.GradientTape() as tape:
            with tf.device('/gpu:0'):
                curr_action_logits, vf_preds = self.model(
                    obs=obs,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    state=state,
                    seq_lens=seq_lens
                )
                mask = tf.transpose(tf.sequence_mask(seq_lens, maxlen=self.config.max_seq_len), [1, 0])
                curr_action_dist = self.model.action_dist(curr_action_logits)
                curr_action_logp = curr_action_dist.logp(action)
                prev_action_dist = self.model.action_dist(action_logits)
                entropy = tf.boolean_mask(curr_action_dist.entropy(), mask)

                logp_ratio = tf.exp(curr_action_logp - action_logp)

                surrogate_loss = tf.minimum(
                    advantages * logp_ratio,
                    advantages
                    * tf.clip_by_value(
                        logp_ratio,
                        1 - self.policy_config.ppo_clip,
                        1 + self.policy_config.ppo_clip,
                    ),
                )

                policy_loss = -tf.reduce_mean(tf.boolean_mask(surrogate_loss, mask))

                critic_loss = self.model.critic_loss(vf_targets)
                critic_loss_clipped = tf.clip_by_value(
                    critic_loss,
                    0,
                    self.policy_config.vf_clip,
                )
                critic_loss = self.policy_config.baseline_coeff * tf.reduce_mean(tf.boolean_mask(critic_loss_clipped, mask))
                mean_entropy = tf.reduce_mean(entropy)

                if self.policy_config.initial_kl_coeff > 0.0:
                    kl_behavior_to_online = tf.boolean_mask(prev_action_dist.kl(curr_action_logits), mask)
                    mean_kl =  tf.reduce_mean(kl_behavior_to_online)
                    kl_loss = mean_kl * self.kl_coeff

                else:
                    mean_kl = 0.
                    kl_loss = tf.constant(0.0)
                total_loss = (critic_loss + policy_loss - mean_entropy * self.policy_config.entropy_cost + kl_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients, mean_grad_norm = tf.clip_by_global_norm(gradients, self.policy_config.grad_clip)

        self.model.optimiser.apply(gradients, self.model.trainable_variables)

        explained_vf = explained_variance(
            tf.boolean_mask(vf_targets, mask),
            tf.boolean_mask(vf_preds, mask)
        )

        train_metrics = {
            "mean_entropy": mean_entropy,
            "vf_loss": critic_loss,
            "pi_loss": policy_loss,
            "mean_grad_norm": mean_grad_norm,
            "explained_vf": explained_vf,
            "kl": mean_kl,
            "kl_loss": kl_loss,
            "kl_coeff": self.kl_coeff,
            "logp_ratio": tf.reduce_mean(tf.boolean_mask(logp_ratio, mask)),
            "x": vf_targets,
            "y": vf_preds

        }

        train_metrics.update(self.model.get_metrics())

        return train_metrics
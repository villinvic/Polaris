import time
from typing import Dict

import tree

from .parametrised import ParametrisedPolicy
from polaris import SampleBatch

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from .popart import Popart
from polaris.models.utils.misc import explained_variance
from polaris.models.utils.vtrace import compute_vtrace



def get_annealed_top_half_exp_advantages(advantages, temperature):

    # TODO: filter top_half
    # if this gets clipped, then the grad does not prop right?
    return tf.exp(tf.minimum(1.4e1, advantages / temperature))

def temperature_loss(temperature, eps_temperature, annealed_top_half_exp_advantage):
    """

    :param temperature:  temperature
    :param eps_temperature:  eps_tempirature
    :param annealed_top_half_exp_advantage:  normalised advantage, should be stop_gradiented
    :return: temperature loss

    """
    return temperature * (eps_temperature + tf.math.log(tf.reduce_mean(
        annealed_top_half_exp_advantage
    )))

def trust_region_loss(trust_region_coeff, trust_region_eps, kl_target_to_online):
    """
    Whole batch of data should be used here
    :param trust_region_coeff: trust_region_coeff
    :param trust_region_eps:  trust_region_eps
    :param kl_target_to_online:  kl divergence between online version and behaviour distribution
    :return: trust region loss
    """
    return tf.reduce_mean(trust_region_coeff * (trust_region_eps - tf.stop_gradient(kl_target_to_online)) +
                          tf.stop_gradient(trust_region_coeff) kl_target_to_online)

def policy_loss(top_half_online_logp, annealed_top_half_exp_advantage):
    psi = tf.stop_gradient(annealed_top_half_exp_advantage / tf.reduce_sum(annealed_top_half_exp_advantage))
    return - tf.reduce_sum(psi * top_half_online_logp)
class VMPO(ParametrisedPolicy):
    # TODO: take care of the loss, tophalf adv, not much more i think

    def __init__(
            self,
            *args,
            online=False,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.popart_module = Popart(
            learning_rate=self.policy_config.popart_lr,
            std_clip=self.policy_config.popart_std_clip
        )

        self.is_online = online
        if self.is_online:
            self.offline_policy = VMPO(*args, online=False, **kwargs)

        self.trust_region_coeff = tf.Variable(self.policy_config.initial_trust_region_coeff, dtype=tf.float32, constraint=tf.relu)
        self.temperature = tf.Variable(self.policy_config.initial_temperature, dtype=tf.float32, constraint=tf.relu)


        if self.version % self.policy_config.target_update_freq % 0:
            self.update_offline_weights()


    def init_model(self):
        super().init_model()
        self.offline_policy.init_model()
        # Copy offline polocy
        self.update_offline_weights()


    def get_online_weights(self):
        return {v.name: v.numpy()
                for v in self.model.trainable_variables}

    def get_weights(self) -> Dict[str, np.ndarray]:
        # Passes the weights of the offline policy
        if self.is_online:
            p = self.offline_policy
        else:
            p = self
        return p.get_weights()

    def update_offline_weights(self):
        self.offline_policy.set_weights(self.get_online_weights())
        self.get_params()


    def train(
            self,
            input_batch: SampleBatch
    ):
        preprocess_start_time = time.time()
        res = super().train(input_batch)

        num_sequences = len(input_batch[SampleBatch.SEQ_LENS])

        # TODO: make this a function, can we do this more efficiently ?
        def make_time_major(v):
            return np.transpose(np.reshape(v, (num_sequences, -1) + v.shape[1:]), (1, 0) + tuple(range(2, 1+len(v.shape))))


        d = dict(input_batch)
        seq_lens = d.pop(SampleBatch.SEQ_LENS)
        time_major_batch = tree.map_structure(make_time_major, d)

        time_major_batch[SampleBatch.OBS] = np.concatenate([time_major_batch[SampleBatch.OBS], time_major_batch[SampleBatch.NEXT_OBS][-1:]], axis=0)
        time_major_batch[SampleBatch.PREV_REWARD] = np.concatenate([time_major_batch[SampleBatch.PREV_REWARD], time_major_batch[SampleBatch.REWARD][-1:]], axis=0)
        time_major_batch[SampleBatch.PREV_ACTION] = np.concatenate([time_major_batch[SampleBatch.PREV_ACTION], time_major_batch[SampleBatch.ACTION][-1:]], axis=0)
        time_major_batch[SampleBatch.STATE] = [state[0] for state in time_major_batch[SampleBatch.STATE]]

        time_major_batch[SampleBatch.SEQ_LENS] = seq_lens

        # Sequence is one step longer if we are not done at timestep T
        time_major_batch[SampleBatch.SEQ_LENS][
            np.logical_and(seq_lens == self.config.max_seq_len,
                           np.logical_not(time_major_batch[SampleBatch.DONE][-1])
                           )] += 1

        nn_train_time = time.time()

        metrics = self._train(
            input_batch=time_major_batch
        )
        popart_update_time = time.time()

        self.popart_module.batch_update(metrics["vtrace_mean"], metrics["vtrace_std"], self.model._value_out)
        metrics["popart"] = self.popart_module.get_metrics()



        metrics.update(**res,
                       preprocess_time_ms=(nn_train_time-preprocess_start_time)*1000.,
                       grad_time_ms=(popart_update_time - nn_train_time) * 1000.,
                       popart_update_ms=(time.time()-popart_update_time)*1000.,
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
                mask_all = tf.transpose(tf.sequence_mask(input_batch[SampleBatch.SEQ_LENS], maxlen=self.config.max_seq_len+1), [1, 0])
                mask = mask_all[:-1]
                action_logits = action_logits[:-1]
                action_dist = self.model.action_dist(action_logits)
                action_logp = action_dist.logp(input_batch[SampleBatch.ACTION])
                kl = tf.boolean_mask(action_dist.kl(input_batch[SampleBatch.ACTION_LOGITS]), mask)
                behavior_logp = input_batch[SampleBatch.ACTION_LOGP]
                entropy = tf.boolean_mask(action_dist.entropy(), mask)
                # TODO: check if mask is helping ?
                all_vf_preds = self.model.value_function() * tf.cast(mask_all, tf.float32)
                vf_preds = all_vf_preds[:-1]
                unnormalised_all_vf_preds = self.popart_module.unnormalise(all_vf_preds)
                unnormalised_vf_pred = unnormalised_all_vf_preds[:-1]
                unnormalised_next_vf_pred = unnormalised_all_vf_preds[1:]
                unnormalised_bootstrap_v = unnormalised_all_vf_preds[-1]
                rhos = tf.exp(action_logp - behavior_logp)
                clipped_rhos= tf.minimum(1.0, rhos, name='cs')

                # TODO: See sequence effect on the vtrace !
                # We are not masking the initial values there... maybe the values must be masked ?
                with tf.device('/cpu:0'):
                    vtrace_returns = compute_vtrace(
                    rewards=input_batch[SampleBatch.REWARD],
                    dones=input_batch[SampleBatch.DONE],
                    values=unnormalised_vf_pred,
                    next_values=unnormalised_next_vf_pred,
                    discount=self.policy_config.discount,
                    clipped_rhos=clipped_rhos,
                    bootstrap_v=unnormalised_bootstrap_v
                    )


                normalised_pg_advantages = tf.boolean_mask(clipped_rhos * (self.popart_module.normalise(vtrace_returns.gpi) - vf_preds), mask)
                values = self.popart_module.normalise(vtrace_returns.gvs)

                #values = input_batch[SampleBatch.REWARD] + next_vf_pred * self.policy_config.discount * (1. - input_batch[SampleBatch.DONE])
                advantage = tf.boolean_mask(values - vf_preds, mask)
                critic_loss =  0.5 * tf.square(advantage)
                policy_loss = - (tf.boolean_mask(action_logp, mask) * tf.stop_gradient(normalised_pg_advantages) + self.policy_config.entropy_cost * entropy)

                total_loss = tf.reduce_mean(policy_loss + critic_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients = [tf.clip_by_norm(g, self.policy_config.grad_clip)
                 for g in gradients]

        self.model.optimiser.apply_gradients(zip(gradients,self.model.trainable_variables))

        mean_entropy = tf.reduce_mean(entropy)
        vf_loss = tf.reduce_mean(critic_loss)
        pi_loss = tf.reduce_mean(policy_loss)
        total_grad_norm = 0.
        num_params = 0
        for grad in gradients:
            total_grad_norm += tf.reduce_sum(tf.abs(grad))
            num_params += tf.size(grad)
        mean_grad_norm = total_grad_norm / tf.cast(num_params, tf.float32)
        explained_vf = explained_variance(
            tf.reshape(tf.boolean_mask(values, mask), [-1]),
            tf.reshape(tf.boolean_mask(vf_preds, mask), [-1]),
        )

        return {
            "mean_entropy": mean_entropy,
            "vf_loss": vf_loss,
            "pi_loss": pi_loss,
            "mean_grad_norm": mean_grad_norm,
            "explained_vf": explained_vf,
            "vtrace_mean": tf.reduce_mean(tf.boolean_mask(vtrace_returns.gvs, mask)),
            "vtrace_std": tf.math.reduce_std(tf.boolean_mask(vtrace_returns.gvs, mask)),
            "kl_divergence": tf.reduce_mean(kl),
            "rhos": tf.reduce_mean(rhos)
        }
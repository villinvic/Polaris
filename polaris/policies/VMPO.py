import time

import tree

from .parametrised import ParametrisedPolicy
from polaris.experience import SampleBatch

import numpy as np
import tensorflow as tf

from polaris.policies.utils.return_based_scaling import ReturnBasedScaling

tf.compat.v1.enable_eager_execution()

from polaris.policies.utils.misc import explained_variance
from polaris.policies.utils.vtrace import compute_vtrace



def get_annealed_softmax_advantages(top_half_advantages, temperature, top_half_clipped_rhos):
    exp_adv = tf.exp(tf.minimum(top_half_advantages / temperature, 20.)) * top_half_clipped_rhos
    return tf.stop_gradient(
        exp_adv / tf.reduce_sum(exp_adv)
    )

def compute_temperature_loss(temperature, temperature_eps, temperature_kl):
    """

    :param temperature:  temperature
    :param temperature_eps:  eps_temperature
    :param temperature_kl:  temperature_kl
    :return: temperature loss

    """
    return temperature * tf.stop_gradient(temperature_eps - temperature_kl)

def compute_trust_region_loss(trust_region_coeff, trust_region_eps, kl_offline_to_online):
    """
    Whole batch of data should be used here
    :param trust_region_coeff: trust_region_coeff
    :param trust_region_eps:  trust_region_eps
    :param kl_offline_to_online:  kl divergence between online version and behaviour distribution
    :return: trust region loss
    """
    return tf.reduce_mean(trust_region_coeff * (trust_region_eps - tf.stop_gradient(kl_offline_to_online)) +
                          tf.stop_gradient(trust_region_coeff) * kl_offline_to_online)

def compute_policy_loss(top_half_online_logp, annealed_top_half_exp_advantage):
    #psi = tf.stop_gradient(annealed_top_half_exp_advantage / tf.reduce_sum(annealed_top_half_exp_advantage))
    return - tf.reduce_sum(top_half_online_logp * annealed_top_half_exp_advantage)

class VMPO(ParametrisedPolicy):
    # TODO: take care of the loss, tophalf adv, not much more i think

    def __init__(
            self,
            *args,
            is_online=False,
            **kwargs
    ):
        self.is_online = is_online
        if self.is_online:
            self.offline_policy = VMPO(*args, online=False, **kwargs)
        else:
            self.offline_policy = None

        super().__init__(
            *args,
            **kwargs
        )

        self.return_based_scaling = ReturnBasedScaling(
            learning_rate=self.policy_config.popart_lr,
            std_clip=self.policy_config.popart_std_clip
        )

        alpha_range = (1e-6, 1e6)

        def trust_region_constraint(v):
            return tf.clip_by_value(
                v,
                *[tf.math.log(c) for c in alpha_range])

        def temperature_constraint(v):
            return tf.clip_by_value(
                v,
                *[tf.math.log(c) / self.policy_config.temperature_speed for c in alpha_range])

        self.trust_region_log_coeff = tf.Variable(
            tf.math.log(self.policy_config.initial_trust_region_coeff),
            constraint=trust_region_constraint,
            trainable=True,
            dtype=tf.float32)

        self.log_temperature = tf.Variable(
            tf.math.log(self.policy_config.initial_temperature)/self.policy_config.temperature_speed,
            constraint=temperature_constraint,
            trainable=True,
            dtype=tf.float32)

    def init_model(self):
        super().init_model()
        if self.offline_policy is not None:
            self.offline_policy.init_model()
            # set weights to offline policy
            self.update_offline_model()


    # def get_online_weights(self):
    #     return {v.name: v.numpy()
    #             for v in self.model.trainable_variables}

    # def get_weights(self) -> Dict[str, np.ndarray]:
    #     # Passes the weights of the offline policy
    #     if self.is_online:
    #         return self.offline_policy.get_weights()
    #     else:
    #         return super().get_weights()
        
    def get_params(self, online=False):
        if not online and self.is_online:
            return self.offline_policy.get_params()
        else:
            return super().get_params()

    def update_offline_model(self):
        self.offline_policy.setup(self.get_params(online=True))

    def setup(self, policy_params: "PolicyParams"):
        super().setup(policy_params)
        if self.is_online:
            self.offline_policy.setup(policy_params)
        return self


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

        #self.popart_module.batch_update(metrics["vtrace_mean"], metrics["vtrace_std"], self.model._value_out)
        self.return_based_scaling.batch_update(time_major_batch[SampleBatch.REWARD], metrics.pop("returns"))
        metrics["return_based_scaling"] = self.return_based_scaling.get_metrics()

        if self.version % self.policy_config.target_update_freq == 0:
            self.update_offline_model()

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
                action_logits, _ = self.model(
                    input_batch
                )
                offline_logits, _ = self.offline_policy.model(
                    input_batch
                )

                mask_all = tf.transpose(tf.sequence_mask(input_batch[SampleBatch.SEQ_LENS], maxlen=self.config.max_seq_len+1), [1, 0])
                mask = mask_all[:-1]
                action_logits = action_logits[:-1]
                action_dist = self.model.action_dist(action_logits)
                action_logp = action_dist.logp(input_batch[SampleBatch.ACTION])

                offline_action_dist = self.model.action_dist(offline_logits[:-1])
                offline_action_logp = tf.stop_gradient(offline_action_dist.logp(input_batch[SampleBatch.ACTION]))

                behavior_logp = input_batch[SampleBatch.ACTION_LOGP]
                entropy = tf.boolean_mask(action_dist.entropy(), mask)
                # TODO: check if mask is helping ?
                all_vf_preds = self.model.value_function()
                vf_preds = all_vf_preds[:-1]
                next_vf_pred = all_vf_preds[1:]
                bootstrap_v = all_vf_preds[-1]
                # unnormalised_all_vf_preds = self.popart_module.unnormalise(all_vf_preds)
                # unnormalised_vf_pred = unnormalised_all_vf_preds[:-1]
                # unnormalised_next_vf_pred = unnormalised_all_vf_preds[1:]
                # unnormalised_bootstrap_v = unnormalised_all_vf_preds[-1]
                rhos = tf.exp(offline_action_logp - behavior_logp)
                # no need to stop the gradient here
                clipped_rhos= tf.stop_gradient(tf.minimum(1.0, rhos, name='cs'))

                # TODO: See sequence effect on the vtrace !
                # We are not masking the initial values there... maybe the values must be masked ?
                with tf.device('/cpu:0'):
                    vtrace_returns = compute_vtrace(
                    rewards=input_batch[SampleBatch.REWARD],
                    dones=input_batch[SampleBatch.DONE],
                    values=vf_preds,
                    next_values=next_vf_pred,
                    discount=self.policy_config.discount,
                    clipped_rhos=clipped_rhos,
                    bootstrap_v=bootstrap_v,
                    mask=mask,
                    )

                # boolean mask flattens already
                rewards = tf.boolean_mask(input_batch[SampleBatch.REWARD], mask)
                gvs = tf.boolean_mask(vtrace_returns.gvs, mask)
                gpi = tf.boolean_mask(vtrace_returns.gpi, mask)
                clipped_rhos = tf.boolean_mask(clipped_rhos, mask)
                vf_preds = tf.boolean_mask(vf_preds, mask)
                online_action_logp = tf.boolean_mask(action_logp, mask)


                batch_sigma = tf.math.sqrt(
                    tf.math.square(tf.math.reduce_std(rewards))
                    +tf.reduce_mean(tf.math.square(gpi))
                )
                #normalised_pg_advantages = tf.boolean_mask(clipped_rhos * (self.popart_module.normalise(vtrace_returns.gpi) - tf.stop_gradient(vf_preds)), mask)
                normalised_pg_advantages = self.return_based_scaling.normalise(clipped_rhos * (gpi - tf.stop_gradient(vf_preds)),
                                                                               batch_sigma=batch_sigma)
                #values = self.popart_module.normalise(vtrace_returns.gvs)
                #values = self.return_based_scaling.normalise(vtrace_returns.gvs)

                advantage =  self.return_based_scaling.normalise(gvs - vf_preds,
                                                                 batch_sigma=batch_sigma)
                critic_loss =  0.5 * tf.reduce_mean(tf.square(advantage))

                num_samples = tf.size(normalised_pg_advantages)

                k = tf.cast(self.policy_config.top_sample_frac * tf.cast(num_samples, tf.float32), tf.int32)
                top_half_normalised_pg_advantages, top_half_indices = tf.math.top_k(normalised_pg_advantages, k=k, sorted=False)

                top_half_clipped_rhos = tf.gather(clipped_rhos, top_half_indices, batch_dims=-1)
                temperature = tf.exp(self.log_temperature * self.policy_config.temperature_speed)
                annealed_top_half_exp_advantages = get_annealed_softmax_advantages(
                    top_half_normalised_pg_advantages,
                    temperature,
                    top_half_clipped_rhos,
                )

                temperature_kl = tf.math.log(
                    tf.reduce_mean(tf.exp(tf.minimum(top_half_normalised_pg_advantages / temperature, 10.)) * top_half_clipped_rhos)
                )

                temperature_loss = compute_temperature_loss(
                    temperature=temperature,
                    temperature_eps=self.policy_config.temperature_eps,
                    temperature_kl=temperature_kl
                )

                kl_offline_to_online = tf.stop_gradient(tf.boolean_mask(offline_action_dist.kl(action_logits), mask))

                trust_region_coeff = tf.exp(self.trust_region_log_coeff)
                trust_region_loss = compute_trust_region_loss(
                    trust_region_coeff=trust_region_coeff,
                    trust_region_eps=self.policy_config.trust_region_eps,
                    kl_offline_to_online=kl_offline_to_online
                )

                entropy_loss = - self.policy_config.entropy_cost * tf.reduce_mean(entropy)

                top_half_online_action_logp = tf.gather(online_action_logp, top_half_indices, batch_dims=-1)

                policy_loss = compute_policy_loss(
                    top_half_online_logp=top_half_online_action_logp,
                    annealed_top_half_exp_advantage=annealed_top_half_exp_advantages
                )

                total_loss = critic_loss + policy_loss + entropy_loss + trust_region_loss + temperature_loss

        vars = self.model.trainable_variables + (self.log_temperature, self.trust_region_log_coeff)
        gradients = tape.gradient(total_loss, vars)
        #tf.clip_by_norm(tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
        gradients = [tf.clip_by_norm(g, self.policy_config.grad_clip)
                 for g in gradients]

        self.model.optimiser.apply_gradients(zip(gradients,vars))

        mean_entropy = tf.reduce_mean(entropy)
        total_grad_norm = 0.
        num_params = 0
        for grad in gradients:
            total_grad_norm += tf.reduce_sum(tf.abs(grad))
            num_params += tf.size(grad)
        mean_grad_norm = total_grad_norm / tf.cast(num_params, tf.float32)
        explained_vf = explained_variance(
            gvs,
            vf_preds
        )

        return {
            "mean_entropy": mean_entropy,
            "vf_loss": critic_loss,
            "pi_loss": policy_loss,
            "temp_loss": temperature_loss,
            "trust_region_loss": trust_region_loss,
            "max_pg_adventage": tf.reduce_max(normalised_pg_advantages / temperature),
            "temperature": temperature,
            "trust_region_coeff": trust_region_coeff,
            "mean_grad_norm": mean_grad_norm,
            "explained_vf": explained_vf,
            "vtrace_mean": tf.reduce_mean(tf.boolean_mask(vtrace_returns.gvs, mask)),
            "vtrace_std": tf.math.reduce_std(tf.boolean_mask(vtrace_returns.gvs, mask)),
            "offline_to_online_kl": tf.reduce_mean(kl_offline_to_online),
            "rhos": tf.reduce_mean(rhos),
            "temperature_kl": temperature_kl,
            "returns": vtrace_returns.gvs
        }
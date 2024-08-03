import time

import tree

from .parametrised import ParametrisedPolicy
from polaris.experience import SampleBatch

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from polaris.policies.utils.popart import Popart
from polaris.policies.utils.misc import explained_variance
from polaris.policies.utils.vtrace import compute_vtrace
class IMPALA(ParametrisedPolicy):

    def __init__(
            self,
            *args,
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

        def add_last_timestep(v1, v2):
            return np.concatenate([v1, v2[-1:]], axis=0)


        time_major_batch[SampleBatch.OBS] = tree.map_structure(
            add_last_timestep,
            time_major_batch[SampleBatch.OBS], time_major_batch[SampleBatch.NEXT_OBS]
        )

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

        #self.return_based_scaling.batch_update(metrics.pop("masked_rewards"), metrics.pop("returns"))
        #metrics["return_based_scaling"] = self.return_based_scaling.get_metrics()

        self.popart_module.batch_update(metrics["vtrace_mean"], metrics["vtrace_std"], value_out=self.model._value_out)
        metrics["popart"] = self.popart_module.get_metrics()

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
                (action_logits, state), all_vf_preds = self.model(
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
                # TODO: check on VMPO for how we did masking
                vf_preds = all_vf_preds[:-1]
                unnormalised_all_vf_preds = self.popart_module.unnormalise(all_vf_preds)
                unnormalised_vf_pred = unnormalised_all_vf_preds[:-1]
                unnormalised_next_vf_pred = unnormalised_all_vf_preds[1:]
                unnormalised_bootstrap_v = unnormalised_all_vf_preds[-1]
                rhos = tf.exp(action_logp - behavior_logp)
                clipped_rhos= tf.stop_gradient(tf.minimum(1.0, rhos, name='cs'))

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
                    bootstrap_v=unnormalised_bootstrap_v,
                    mask=mask
                    )


                normalised_pg_advantages = tf.boolean_mask(clipped_rhos * (self.popart_module.normalise(vtrace_returns.gpi) - vf_preds), mask)
                values = self.popart_module.normalise(vtrace_returns.gvs)

                #values = input_batch[SampleBatch.REWARD] + next_vf_pred * self.policy_config.discount * (1. - input_batch[SampleBatch.DONE])
                advantage = tf.boolean_mask(values - vf_preds, mask)
                critic_loss =  0.5 * tf.reduce_mean(tf.square(advantage))
                policy_loss = - tf.reduce_mean(tf.boolean_mask(action_logp, mask) * tf.stop_gradient(normalised_pg_advantages))

                mean_entropy = tf.reduce_mean(entropy)
                entropy_loss = - mean_entropy * self.policy_config.entropy_cost

                total_loss = critic_loss + policy_loss + entropy_loss

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.policy_config.grad_clip)
        mean_grad_norm = tf.linalg.global_norm(gradients)

        self.model.optimiser.apply(gradients,self.model.trainable_variables)
        explained_vf = explained_variance(
           values,
            vf_preds,
        )

        return {
            "mean_entropy": mean_entropy,
            "vf_loss": critic_loss,
            "pi_loss": policy_loss,
            "mean_grad_norm": mean_grad_norm,
            "explained_vf": explained_vf,
            "vtrace_mean": tf.reduce_mean(tf.boolean_mask(vtrace_returns.gvs, mask)),
            "vtrace_std": tf.math.reduce_std(tf.boolean_mask(vtrace_returns.gvs, mask)),
            "kl_divergence": tf.reduce_mean(kl),
            "rhos": tf.reduce_mean(rhos)
        }
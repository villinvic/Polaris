from typing import NamedTuple

import tensorflow as tf


class VtraceReturns(NamedTuple):
    gvs: tf.Tensor
    gpi: tf.Tensor
def compute_vtrace(
        rewards,
        dones,
        values,
        next_values,
        discount,
        clipped_rhos,
        bootstrap_v,
        mask

) -> VtraceReturns:
    """
    :param rewards: rewards, of shape [B, T]
    :param dones: episode masks, of shape [B, T]
    :param values: estimated values, of shape [B, T]
    :param next_values: estimated next_values, of shape [B, T]
    :param discount: discount factor
    :param clipped_rhos: clipped fraction between the online and behavior distributions
    :param bootstrap_v: estimated value at time T+1
    :param mask: sequence mask
    :return: vtrace
    """

    def bellman(future, present):
        discount_t, c_t, delta_t = present
        return delta_t + discount_t * c_t * future

    discounts = (1.-tf.cast(dones, tf.float32)) * tf.cast(mask, tf.float32) * discount
    deltas = clipped_rhos * (rewards + discounts * next_values - values)

    sequence = (
        discounts,
        clipped_rhos,
        deltas,
    )
    vs_minus_v_xs = tf.scan(
        bellman,
        sequence,
        tf.zeros_like(bootstrap_v),
        parallel_iterations=1,
        reverse=True,
        name = 'scan'
    )

    vs = tf.add(vs_minus_v_xs, values, name='vs')
    vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_v, 0)], axis=0)

    gpi = rewards + discounts * vs_t_plus_1

    return VtraceReturns(gvs=tf.stop_gradient(vs), gpi=tf.stop_gradient(gpi))
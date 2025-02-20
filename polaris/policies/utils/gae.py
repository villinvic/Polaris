import numpy as np
import scipy
import tensorflow as tf
import tree

from polaris.experience.episode import batchify_input
from polaris.experience import SampleBatch
from polaris.experience.sampling import concat_sample_batches
from polaris.policies import Policy


def compute_gae(rewards, dones, values,  discount, gae_lambda, bootstrap_v):
    """
    :param rewards: rewards, of shape [B, T]
    :param dones: episode masks, of shape [B, T]
    :param values: estimated values, of shape [B, T-1]
    :param discount: discount factor
    :param gae_lambda:  gae lambda
    :param bootstrap_v: estimated value at NEXT_OBS|NEXT_STATE at the last timestep
    :return:
    """

    def bellman(future, present):
        v, r, d = present
        return (1. - gae_lambda) * v + gae_lambda * (r + (1.-d) * discount * future)

    reversed_sequence = [tf.reverse(t, [0]) for t in [
        tf.transpose(values, [1, 0]),
        tf.transpose(rewards, [1, 0]),
        tf.transpose(tf.cast(dones, tf.float32), [1,0])]
                         ]
    returns = tf.scan(bellman, reversed_sequence, bootstrap_v)
    returns = tf.reverse(returns, [0])

    return tf.stop_gradient(tf.transpose(returns, [1, 0]))


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma: The discount factor gamma.

    Returns:
        The sequence containing the discounted cumulative sums
        for each individual reward in `x` till the end of the trajectory.
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


# from RLlib
def compute_advantages(
    batch: SampleBatch,
    last_r: float,
    gamma: float = 0.9,
    lambda_: float = 1.0,
):
    """Given a rollout, compute its value targets and the advantages.

    Args:
        rollout: SampleBatch of a single trajectory.
        last_r: Value estimation for last observation.
        gamma: Discount factor.
        lambda_: Parameter for GAE.

    Returns:
        SampleBatch with experience from rollout and processed rewards.
    """
    rewards = batch[SampleBatch.REWARD]
    vf_preds = batch[SampleBatch.VALUES]
    not_dones = 1.-np.float32(batch[SampleBatch.DONE])

    next_not_dones = np.concatenate([np.array([1.]), not_dones[:-1]])

    vpred_t = np.concatenate([vf_preds, np.array([last_r])])
    delta_t = rewards + gamma * not_dones * vpred_t[1:] - vpred_t[:-1]

    # TODO: we are one step off
    delta_t *= next_not_dones

    # This formula for the advantage comes from:
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    batch[SampleBatch.ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
    batch[SampleBatch.VF_TARGETS] = (
        (batch[SampleBatch.ADVANTAGES] + vf_preds) * next_not_dones
    ).astype(np.float32)

    batch[SampleBatch.ADVANTAGES] = batch[SampleBatch.ADVANTAGES].astype(
        np.float32
    )
    return batch


def compute_bootstrap_value(sample_batch: SampleBatch, policy: Policy) -> SampleBatch:
    """Performs a value function computation at the end of a trajectory.
    """

    # Trajectory is actually complete -> last r=0.0.
    if SampleBatch.VALUES not in sample_batch:
        # attempt to compute the values
        raise NotImplementedError("we have yet to implement this !")
        sample_batch[SampleBatch.VALUES] = values



    if sample_batch[SampleBatch.DONE][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        last_r = policy.compute_single_action_with_extras(
            **batchify_input(
            obs= tree.map_structure(lambda v: v[-1],sample_batch[SampleBatch.NEXT_OBS]),
            prev_action=sample_batch[SampleBatch.ACTION][-1],
            prev_reward=sample_batch[SampleBatch.REWARD][-1],
            state=sample_batch[SampleBatch.NEXT_STATE]
            )
        )[2][SampleBatch.VALUES]

    sample_batch[SampleBatch.BOOTSTRAP_VALUE] = last_r
    return sample_batch


def compute_gae_for_sample_batch(
    policy: Policy,
    sample_batch: SampleBatch,
) -> SampleBatch:

    if SampleBatch.BOOTSTRAP_VALUE not in sample_batch:
        sample_batch = compute_bootstrap_value(sample_batch, policy)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.

    batch = compute_advantages(
        batch=sample_batch,
        last_r=sample_batch[SampleBatch.BOOTSTRAP_VALUE],
        gamma=policy.policy_config.discount,
        lambda_=policy.policy_config.gae_lambda,
    )

    #if batch[SampleBatch.DONE][-1]:
    #    print(batch[SampleBatch.ADVANTAGES])

    return batch



def batched_gae(
        sample_batch: SampleBatch,
        policy: Policy,
):
    """
    Computes and adds entries for advantages and vf_targets in the sample batch.

    Returns:
        dict: Updated sample_batch with 'advantages' and 'vf_targets' added.
    """

    gamma = policy.policy_config.discount
    gae_lambda = policy.policy_config.gae_lambda


    values = sample_batch[SampleBatch.VALUES]
    rewards = sample_batch[SampleBatch.REWARD]
    dones = sample_batch[SampleBatch.DONE]
    bootstrapped_values = sample_batch[SampleBatch.BOOTSTRAP_VALUE]

    values = np.vstack([values, bootstrapped_values[np.newaxis, :]])

    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    deltas_flipped = deltas[::-1]

    filter_coeff = [1]
    gain = gamma * gae_lambda

    advantages_flipped = scipy.signal.lfilter(filter_coeff, [1, -gain], deltas_flipped, axis=0)

    # Reverse advantages back to original order
    advantages = advantages_flipped[::-1]

    # Compute value function targets
    vf_targets = advantages + sample_batch[SampleBatch.VALUES]

    # Update sample_batch
    sample_batch[SampleBatch.ADVANTAGES] = advantages.astype(np.float32)
    sample_batch[SampleBatch.VF_TARGETS] = vf_targets.astype(np.float32)

    return sample_batch


if __name__ == '__main__':
    q = SampleBatch(20)
    for i in range(20):

        q.push({SampleBatch.OBS: np.array([1, 2]), SampleBatch.ACTION: 2, SampleBatch.REWARD: np.float32(np.random.randint(5)), SampleBatch.DONE: np.random.random() < 0.1})

    q = concat_sample_batches([q,q])
    rewards = q[SampleBatch.REWARD]
    dones = q[SampleBatch.DONE]
    print(rewards, dones)
    returns = compute_gae(rewards, dones, np.zeros((2, 20), dtype=np.float32), 0.99, 1., np.array([0., 0.], np.float32))
    print(returns.numpy())

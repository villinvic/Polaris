import numpy as np
import tensorflow as tf
from polaris import SampleBatch
from polaris.experience.sampling import concat_sample_batches


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

import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

class GaussianDistribution:

    def __init__(self, logits):
        self.means, self.log_stds = tf.split(logits, 2, axis=-1)
        self.stds = tf.exp(self.log_stds)
        self.dist = None
        self.pi = tf.constant(np.pi)


    def _compute_dist(self):
        if self.dist is None:
            self.dist = tfp.distributions.Normal(loc=self.means, scale=self.stds)


    def logp(self, x):
        self._compute_dist()
        return self.dist.log_prob(x)

    def sample(self):
        self._compute_dist()
        return self.dist.sample()

    def entropy(self):
        self._compute_dist()
        return self.dist.entropy()





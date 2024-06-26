import tensorflow_probability as tfp
import tensorflow as tf

class CategoricalDistribution:

    def __init__(self, logits):
        self.logits = logits

        self.dist = None


    def _compute_dist(self):
        if self.dist is None:
            self.dist = tfp.distributions.Categorical(logits=self.logits)

    def logp(self, actions):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=tf.cast(actions, tf.int32),
        )

    def sample(self):
        self._compute_dist()
        return self.dist.sample()

    def entropy(self):
        self._compute_dist()
        return self.dist.entropy()

    def kl(self, other):
        self._compute_dist()
        return self.dist.kl_divergence(tfp.distributions.Categorical(logits=other))





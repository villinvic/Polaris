import tensorflow_probability as tfp
import tensorflow as tf

class EpsilonCategorical:
    def __init__(self, logits, epsilon=1e-2):
        self.logits = logits
        self.logits -= tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
        self.probs = tf.nn.softmax(self.logits)
        self.epsilon = epsilon
        self.dist = None

    def _compute_dist(self):
        if self.dist is None:
            probs = self.epsilon / tf.cast(tf.shape(self.probs)[-1], tf.float32) + (1.-self.epsilon) * self.probs
            logits = tf.math.log(probs)
            self.dist = tfp.distributions.Categorical(logits=logits)

    def logp(self, actions):
        self._compute_dist()
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.dist.logits,
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

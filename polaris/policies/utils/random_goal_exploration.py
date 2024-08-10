import tensorflow as tf

class RandomGoalExploration:

    def __init__(self, config):
        self.goal_vector = None
        self.config = config
        self.mean_exploration_rewards = 0.

    def randomise_goals(self, model, version):
        if version % self.config.goal_randomisation_freq == 0:
            model._goal_vector.assign(tf.nn.softmax(tf.random.normal((1, 1, self.config.random_embedding_dims[-1]), stddev=3.0,
)))


    def compute_rewards(self, model):

        r = tf.stop_gradient(tf.reduce_sum(model._random_embeddings * model._goal_vector, axis=-1))
        self.mean_exploration_rewards = tf.reduce_mean(r)

        return r

    def get_metrics(self):
        return {
            "random_goal_exploration_rewards": self.mean_exploration_rewards
        }


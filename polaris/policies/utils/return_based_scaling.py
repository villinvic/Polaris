import numpy as np
import tensorflow as tf
import sonnet as snt

class ReturnBasedScaling:

    def __init__(self, init_var=1., learning_rate=0.2, std_clip=1e-2):
        # https://arxiv.org/pdf/2105.05347"
        # TODO: save this into checkpoint

        self.init_std = init_var
        self.lr = learning_rate
        self.std_clip = std_clip

        self.squared_return_mean = 0.
        self.rewards_var = 1.
        self.rewards_mean = 0.
        self.curr_lr = 0.9

        #self.c = 0.


    def batch_update(self, rewards, returns):
        # next_c = self.c + 1
        # rewards_mean = tf.reduce_mean(rewards)
        # new_reward_mean = 1 / next_c * rewards_mean + self.c/next_c * self.rewards_mean
        # self.rewards_var = 1 / next_c * np.var(rewards) + self.c/next_c * self.rewards_var + self.c / next_c**2 * np.square(rewards_mean - self.rewards_mean)
        # self.rewards_mean = new_reward_mean
        # self.squared_return_mean = 1 / next_c * np.mean(np.square(returns)) + self.c/next_c  * self.squared_return_mean
        # self.c = next_c

        lr = self.curr_lr
        mlr = 1.-lr

        rewards_mean = tf.reduce_mean(rewards)
        new_reward_mean = lr * rewards_mean + mlr * self.rewards_mean

        self.rewards_var = lr * np.var(rewards) + mlr * self.rewards_var + lr * mlr * np.square(rewards_mean - self.rewards_mean)
        self.rewards_mean = new_reward_mean
        self.squared_return_mean = lr * np.mean(np.square(returns)) + mlr * self.squared_return_mean

        self.curr_lr = np.maximum((1-self.lr) * self.curr_lr, self.lr)

    def __call__(self):
        return tf.maximum(self.std_clip, tf.sqrt(self.rewards_var + self.squared_return_mean))
    def normalise(self, v, batch_sigma):
        return v / tf.maximum(self(), batch_sigma)

    def unnormalise(self, v):
        return v * self.std + self.mean

    def get_metrics(self):
        return {
            "squared_return_mean": self.squared_return_mean,
            "rewards_var": self.rewards_var,
            "rewards_mean": self.rewards_mean,
            "clipping_scale": self(),
            "learning_rate": self.curr_lr
        }


import numpy as np
import tensorflow as tf
import sonnet as snt

class Popart:

    def __init__(self, init_std=1., learning_rate=1e-2, std_clip=1e-2):
        self.init_std = init_std
        self.lr = learning_rate
        self.mlr = 1.-self.lr
        self.lrmlr = self.lr * self.mlr
        self.std_clip = std_clip


        self.mean = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.std = tf.Variable(init_std, dtype=tf.float32, trainable=False)


    def batch_update(self, mean, std, value_out: snt.Linear):

        new_mean = self.mean * self.mlr + mean * self.lr

        new_var = tf.square(self.std) * self.mlr + tf.square(std) * self.lr + tf.square(self.mean - mean) * self.lrmlr
        new_std = tf.clip_by_value(tf.sqrt(new_var), self.std_clip, 1e6)

        value_out.b.assign(
            (value_out.b * self.std + self.mean - new_mean) / new_std
        )
        value_out.w.assign(
            value_out.w * self.std / new_std
        )

        self.mean.assign(new_mean)
        self.std.assign(new_std) #+ tf.square(self.mean-mean) * self.lrmlr)

    def normalise(self, v):
        return (v - self.mean) / self.std

    def unnormalise(self, v):
        return v * self.std + self.mean

    def get_metrics(self):
        return {
            "mean": self.mean.numpy(),
            "std": self.std.numpy()
        }

    def get_weights(self):
        return self.get_metrics()


    def set_weights(self, w):

        self.mean.assign(w["mean"])
        self.std.assign(w["std"])




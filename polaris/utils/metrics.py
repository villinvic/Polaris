from collections import defaultdict

from ray.experimental.tf_utils import tf

from .paths import PathManager
import numpy as np

GlobalCounter = defaultdict(int)

class Metric:
    def __init__(
            self,
            name="metric",
            init_value=np.nan,
            smoothing=1.0,
            n_init=1,
            save_history=False,
    ):
        """
        :param name: name of the metric
        :param init_value: to which value the metric should be initialised, None (default) will initialise to the first value observed
        :param smoothing: Smoothing factor
        :param n_init: How many samples before estimating the first next value ?
        """
        assert 0 <= smoothing < 1
        self.name = name
        self.lr = smoothing
        self._v = init_value
        self.n_init = n_init

        self.init_value = init_value
        self.init_count = 0
        self.history = [] if save_history else None

    def update(self, value, n_samples=1):
        """

        :param value: next value
        :param n_samples: how many samples is this value averaged over ?
        :return:
        """

        if not np.any(np.isnan(value)):
            if np.any(np.isnan(self._v)):
                next_count = self.init_count + n_samples
                if self.init_value is None:
                    self.init_value = value
                else:

                    self.init_value = value * n_samples / next_count + self.init_value * self.init_count / next_count
                self.init_count = next_count
                if self.init_count >= self.n_init:
                    self.set(self.init_value)
            else:
                lr = np.maximum(1. - self.lr * n_samples, 0.)
                self._v = (1. - lr) * self._v + lr * value
                if self.history is not None:
                    if isinstance(self._v, np.ndarray):
                        self.history.append(self._v.copy())
                    else:
                        self.history.append(self._v)

        return self._v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v

    def report(self):
        if isinstance(self._v, np.ndarray):
            tf.summary.scalar(self.name+"_max", np.max(self._v), step=GlobalCounter["step"])
            tf.summary.scalar(self.name + "_min", np.min(self._v), step=GlobalCounter["step"])
            tf.summary.scalar(self.name + "_mean", np.mean(self._v), step=GlobalCounter["step"])
        else:
            tf.summary.scalar(self.name, self._v, step=GlobalCounter["step"])

class MetricBank:

    def __init__(
            self,
            dirname,
            report_freq=3,
            ):
        """

        :param dirname: path to the tensorboard logs.
        :param report_freq: frequency at which the bank reports to tensorboard (alleviates disk memory usage)
        """

        self.metrics = {}
        self.logdir = PathManager.get_tensorboard_logdir(dirname)
        self.report_freq = report_freq

        import tensorflow as tf
        self.writer = tf.summary.create_file_writer(self.logdir)

    def track_metric(self, name: str, init_value=np.nan, smoothing=1.0, n_init=1, save_history=False):
        # TODO : take care of hist, etc
        self.metrics[name] = Metric(name, init_value, smoothing, n_init, save_history)

    def update(self, batch, n_samples=1):
        """
        :param batch: iteration data, data that is tracked by this bank will be updated.
        :param n_samples: how many samples are contained in the batch
        """

        for metric_name, metric in self.metrics.items():
            if metric_name in batch:
                metric.update(batch[metric_name], n_samples=n_samples)

        if GlobalCounter["step"] % self.report_freq == 0:
            if self.writer.as_default():
                for metric in self.metrics.values():
                    metric.report()

    def __del__(self):
        self.writer.close()





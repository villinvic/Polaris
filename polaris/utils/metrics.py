import threading
import time
from collections import defaultdict

from ray.experimental.tf_utils import tf

from .paths import PathManager
import numpy as np


class GlobalVars:
    def __init__(self, factory):
        self.dict = defaultdict(factory)
        self.lock = threading.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            self.dict[key] = value

    def __getitem__(self, item):
        return self.dict[item]

    def get(self):
        return self.dict

class _GlobalCounter(GlobalVars):
    STEP = "step"
    ENV_STEPS = "environment_steps"

    def __init__(self):
        super().__init__(int)

    def incr(self, key):
        self[key] += 1


class _GlobalTimer(GlobalVars):
    PREV_ITERATION = "previous_iteration"
    PREV_FRAMES = "previous_frames"

    def __init__(self):
        super().__init__(float)
        self._dt = defaultdict(float)
        self.startup_time = time.time()

    def __setitem__(self, key, value):
        with self.lock:
            prev = self[key]
            value = value - self.startup_time
            self._dt[key] = value - prev
            self.dict[key] = value

    def dt(self, key):
        return self._dt[key]

GlobalCounter = _GlobalCounter()
GlobalTimer = _GlobalTimer()


class Metric:
    def __init__(
            self,
            name="metric",
            init_value=np.nan,
            smoothing=0.0,
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
            tf.summary.scalar(self.name+"_max", np.max(self._v), step=GlobalCounter[GlobalCounter.ENV_STEPS])
            tf.summary.scalar(self.name + "_min", np.min(self._v), step=GlobalCounter[GlobalCounter.ENV_STEPS])
            tf.summary.scalar(self.name + "_mean", np.mean(self._v), step=GlobalCounter[GlobalCounter.ENV_STEPS])
        else:
            tf.summary.scalar(self.name, self._v, step=GlobalCounter[GlobalCounter.ENV_STEPS])

class Metrics(dict): pass

class MetricBank:

    def __reduce__(self):
        return MetricBank, self.args

    def __init__(
            self,
            dirname,
            report_dir="",
            report_freq=3,
            metrics=Metrics(),
            ):
        """

        :param dirname: path to the tensorboard logs.
        :param report_freq: frequency at which the bank reports to tensorboard (alleviates disk memory usage)
        """

        self.args = (dirname, report_dir, report_freq, metrics)

        self.metrics = Metrics()
        self.logdir = PathManager(base_dir=report_dir).get_tensorboard_logdir(dirname)
        self.report_freq = report_freq
        self.last_report = -1

        import tensorflow as tf
        self.writer = tf.summary.create_file_writer(self.logdir)

    def get(self):
        return self.metrics

    def track_metric(self, name: str, init_value=np.nan, smoothing=0.0, n_init=1, save_history=False):
        # TODO : take care of hist, etc
        self.metrics[name] = Metric(name, init_value, smoothing, n_init, save_history)

    def update(self, batch, n_samples=1, prefix="", smoothing=0.0):
        """
        :param batch: iteration data, data that is tracked by this bank will be updated.
        :param n_samples: how many samples are contained in the batch
        :param prefix: additional prefix for the batch metrics
        :param smoothing: smoothing for new entries
        """

        for metric_name, value in batch:
            if isinstance(metric_name, tuple):
                metric_name = metric_path_name(metric_path=metric_name)
            if value is not None:
                p_metric_name =  prefix + metric_name
                if p_metric_name in self.metrics:
                    self.metrics[p_metric_name].update(value, n_samples=n_samples)
                else:
                    self.metrics[p_metric_name] = Metric(
                        name=p_metric_name,
                        init_value=value,
                        smoothing=smoothing
                    )
    def report(self, print_metrics=False):

        if self.last_report != GlobalCounter["step"] and GlobalCounter["step"] % self.report_freq == 0:
            with self.writer.as_default():
                self.last_report = GlobalCounter["step"]
                if print_metrics:
                    print(f"==================================== Iteration {GlobalCounter['step']} ====================================")
                for name, metric in self.metrics.items():
                    metric.report()
                    if print_metrics:
                        print(f"{name:<55}:\t{metric.get():.5f}")

            self.writer.flush()

    def __del__(self):
        self.writer.close()


def metric_path_name(metric_path):
    return  "/".join(metric_path).replace("'", "").replace(",","").replace(" ", "_")



import threading
import time
from collections import defaultdict

import tree
from polaris.utils import plot_utils

from .paths import PathManager
import numpy as np
import wandb


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
    NUM_EPISODES = "num_episodes"

    def __init__(self):
        super().__init__(int)

    def incr(self, key):
        self[key] += 1


class _GlobalTimer(GlobalVars):
    PREV_ITERATION = "previous_iteration"
    PREV_FRAMES = "previous_frames"

    def __init__(self):
        super().__init__(float)
        self._dt = defaultdict(lambda : 1e9)
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

def merge(args):
    if any(not isinstance(i, dict) for i in args):
        return np.array([args[0]]) if len(args) == 1 else np.array(args)
    d = defaultdict(list)
    for i in args:
        for a, b in i.items():
            d[a].append(b)
    return {a: merge(b) for a, b in d.items()}

def average_dict(data: list):
    return tree.map_structure(
        lambda v: np.mean(v),
        merge(data)
    )


class Metric:
    def __init__(
            self,
            name="metric",
            init_value=None,
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
        if isinstance(init_value, np.ndarray) and init_value.dtype == object:
            init_value = init_value.item()

        assert 0 <= smoothing < 1
        self.name = name
        self.lr = smoothing

        self._v = init_value
        self.n_init = n_init

        self.init_value = init_value
        self.init_count = 0
        self.history = [] if save_history else None

        self.last_update = 0

    def update(self, value, n_samples=1):
        """

        :param value: next value
        :param n_samples: how many samples is this value averaged over ?
        :return:
        """
        # hack for dicts/barplots
        if isinstance(value, np.ndarray) and value.dtype == object:
            value = value.item()


        if value is not None:
            if self._v is None:
                next_count = self.init_count + n_samples
                if self.init_value is None:
                    self.init_value = value
                else:
                    if isinstance(value, dict):
                        # barplot
                        for k in value:
                            self.init_value[k] = value[k] * n_samples / next_count + self.init_value[k] * self.init_count / next_count
                    else:
                        self.init_value = value * n_samples / next_count + self.init_value * self.init_count / next_count
                self.init_count = next_count
                if self.init_count >= self.n_init:
                    self.set(self.init_value)
            else:
                lr = np.maximum(1. - self.lr * n_samples, 0.)
                if isinstance(value, dict) :
                    # barplot
                    for k in value:
                        self._v[k] = (1. - lr) * self._v[k] + lr * value[k]
                else:
                    self._v = (1. - lr) * self._v + lr * value
                if self.history is not None:
                    if isinstance(self._v, np.ndarray):
                        self.history.append(self._v.copy())
                    else:
                        self.history.append(self._v)

        self.last_update = np.int32(GlobalCounter["step"])
        return self._v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v

    def report(self):
        step = np.int32(GlobalCounter[GlobalCounter.ENV_STEPS])
        if isinstance(self._v, dict):
            fig = plot_utils.dict_barplot(self._v, color='skyblue')
            wandb.log({self.name: fig}, step=step)
        if isinstance(self._v, np.ndarray):
            wandb.log({
                self.name+"_max": np.max(self._v),
                self.name + "_min": np.min(self._v),
                self.name + "_mean": np.mean(self._v),

            }, step=step)
            # tf.summary.scalar(self.name+"_max", np.max(self._v), step=GlobalCounter[GlobalCounter.ENV_STEPS])
            # tf.summary.scalar(self.name + "_min", np.min(self._v), step=GlobalCounter[GlobalCounter.ENV_STEPS])
            # tf.summary.scalar(self.name + "_mean", np.mean(self._v), step=GlobalCounter[GlobalCounter.ENV_STEPS])
        else:
            wandb.log(
                {self.name: self._v}, step=step
            )
            #tf.summary.scalar(self.name, self._v, step=GlobalCounter[GlobalCounter.ENV_STEPS])

    def is_old(self):
        return (not hasattr(self, "last_update")) or (GlobalCounter["step"] - self.last_update > 100)

class Metrics(dict): pass

class MetricBank:

    def __reduce__(self):
        return MetricBank, self.args

    def __init__(
            self,
            report_freq=3,
            metrics=Metrics(),
            ):
        """

        :param dirname: path to the tensorboard logs.
        :param report_freq: frequency at which the bank reports to tensorboard (alleviates disk memory usage)
        """

        self.args = (report_freq, metrics)

        self.metrics = Metrics()
        self.report_freq = report_freq
        self.last_report = -1

        #import tensorflow as tf
        #self.writer = tf.summary.create_file_writer(self.logdir)


    def get(self):
        return self.metrics

    def track_metric(self, name: str, init_value=None, smoothing=0.0, n_init=1, save_history=False):
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
                p_metric_name = prefix + metric_name
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
            self.last_report = GlobalCounter["step"]
            if print_metrics:
                print(f"==================================== Iteration {GlobalCounter['step']} ====================================")
            to_delete = []
            for name, metric in self.metrics.items():
                if metric.is_old():
                    to_delete.append(name)
                else:
                    metric.report()
                if print_metrics:
                    print(f"{name:<55}:\t{metric.get():.5f}")
            for k in to_delete:
                del self.metrics[k]



def metric_path_name(metric_path):
    return  "/".join(metric_path).replace("'", "").replace(",","").replace(" ", "_")



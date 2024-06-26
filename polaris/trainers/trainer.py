import importlib
import queue
import threading
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import tree
from ml_collections import ConfigDict

from polaris.experience.episode import EpisodeMetrics, NamedPolicyMetrics
from polaris.experience.worker_set import WorkerSet
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import Policy, PolicyParams
from polaris.experience.matchmaking import RandomMatchmaking
from polaris.experience.sampling import ExperienceQueue, SampleBatch
from polaris.utils.metrics import MetricBank, GlobalCounter, GlobalTimer


class Trainer:
    def __init__(
            self,
            config: ConfigDict,
    ):
        self.config = config
        self.worker_set = WorkerSet(
            config
        )

        # Init environment
        self.env = PolarisEnv.make(self.config.env)

        # We can extend to multiple policy types if really needed, but won't be memory efficient
        policy_params = [
            PolicyParams(**ConfigDict(pi_params)) for pi_params in self.config.policy_params
        ]

        PolicylCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        self.policy_map: Dict[str, Policy] = {
            policy_param.name: PolicylCls(
                name=policy_param.name,
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                config=self.config,
                policy_config=policy_param.config,
            )
            for policy_param in policy_params
        }

        self.params_map: Dict[str, PolicyParams] = {
            name: p.get_params() for name, p in self.policy_map.items()
        }

        self.experience_queue: Dict[str, ExperienceQueue] = {
            policy_name: ExperienceQueue(self.config)
            for policy_name in self.policy_map
        }

        self.matchmaking = RandomMatchmaking(
            agent_ids=self.env.get_agent_ids(),
            policy_params=self.params_map
        )

        self.runnning_jobs = []

        self.metrics = MetricBank(
            dirname=self.config.tensorboard_logdir,
            report_dir=f"polaris_results/{self.config.seed}",
            report_freq=self.config.report_freq
        )

        # self.grad_thread = GradientThread(
        #     env=self.env,
        #     config=self.config
        # )
        # self.grad_lock = self.grad_thread.lock
        # self.grad_thread.start()

    def training_step(self):
        """
        Executes one iteration of the trainer.
        :return: Training iteration results
        """

        t = []
        t.append(time.time())
        GlobalTimer[GlobalTimer.PREV_ITERATION] = time.time()
        iteration_dt = GlobalTimer.dt(GlobalTimer.PREV_ITERATION)

        t.append(time.time())
        n_jobs = self.worker_set.get_num_worker_available()
        jobs = [self.matchmaking.next() for _ in range(n_jobs)]
        t.append(time.time())

        self.runnning_jobs += self.worker_set.push_jobs(jobs)
        experience_metrics = []
        t.append(time.time())
        frames = 0
        env_steps = 0

        experience = self.worker_set.wait(self.runnning_jobs)
        enqueue_time_start = time.time()
        num_batch = 0
        for exp_batch in experience:
            if isinstance(exp_batch, EpisodeMetrics):
                experience_metrics.append(exp_batch)
                env_steps += exp_batch.length
            else: # Experience batch
                num_batch +=1
                owner = self.policy_map[exp_batch.get_owner()]
                if owner.is_recurrent:
                    exp_batch[SampleBatch.SEQ_LENS] = np.array(exp_batch[SampleBatch.SEQ_LENS])
                    exp_batch = exp_batch.pad_and_split_to_sequences()
                else:
                    exp_batch[SampleBatch.SEQ_LENS] = np.array([exp_batch.max_seq_len], dtype=np.int32)
                self.experience_queue[owner.name].push([exp_batch])
                GlobalCounter.incr("batch_count")
                frames += exp_batch.size()
        if num_batch > 0:
            enqueue_time_ms = (time.time() - enqueue_time_start) * 1000.
        else:
            enqueue_time_ms = None

        GlobalCounter[GlobalCounter.ENV_STEPS] += env_steps

        t.append(time.time())
        training_metrics = {}
        for policy_name, policy_queue in self.experience_queue.items():
            if policy_queue.is_ready():
                train_results = self.policy_map[policy_name].train(
                    policy_queue.pull(self.config.train_batch_size)
                )
                training_metrics[f"{policy_name}"] = train_results
                GlobalCounter.incr(GlobalCounter.STEP)
                self.params_map[policy_name] = self.policy_map[policy_name].get_params()

        #grad_thread_out = self.grad_thread.get_metrics()

        # training_metrics = []
        # for data in grad_thread_out:
        #     if isinstance(data, list):
        #         for policy_param in data:
        #             self.params_map[policy_param.name] = policy_param
        #     else:
        #         training_metrics.append(data)


        def mean_metric_batch(b):
            return tree.flatten_with_path(tree.map_structure(
                lambda *samples: np.mean(samples),
                *b
            ))

        if len(training_metrics)> 0:
            training_metrics = mean_metric_batch([training_metrics])
            self.metrics.update(training_metrics, prefix="training/")
        if len(experience_metrics) > 0:
            mean_batched_experience_metrics = mean_metric_batch(experience_metrics)
            self.metrics.update(mean_batched_experience_metrics, prefix="experience/",
                                smoothing=self.config.episode_metrics_smoothing)

        misc_metrcis =  [
                    ("FPS", frames / iteration_dt),
                ] + [
                    (f'{pi}_queue_length', queue.size())
                    for pi, queue in self.experience_queue.items()
                ]
        if enqueue_time_ms is not None:
            misc_metrcis.append(("experience_enqueue_ms", enqueue_time_ms))


        self.metrics.update(
                [
                    ("FPS", frames / iteration_dt),
                ] + [
                    (f'{pi}_queue_length', queue.size())
                    for pi, queue in self.experience_queue.items()
                ]
            , prefix="misc/", smoothing=0.9
        )

        # We should call those only at the report freq...
        self.metrics.update(
            tree.flatten_with_path(GlobalCounter.get()), prefix="counters/", smoothing=0.9
        )
        # self.metrics.update(
        #     tree.flatten_with_path(GlobalCounter), prefix="timers/", smoothing=0.9
        # )

        t.append(time.time())

        #print(np.diff(t))

        #print(GlobalCounter.dict)


    def run(self):
        try:
            while True:
                self.training_step()
                self.metrics.report(print_metrics=True)
        except KeyboardInterrupt:

            print("Caught C^.")
        #self.grad_thread.stop()


class GradientThread(threading.Thread):
    def __init__(
            self,
            env: PolarisEnv,
            #policy_map: Dict[str, Policy],
            config: ConfigDict,
    ):
        threading.Thread.__init__(self)

        self.env = env
        self.config = config

        self.experience_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):

        PolicylCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        ModelCls = getattr(importlib.import_module(self.config.model_path), self.config.model_class)
        policy_params = [
            PolicyParams(**pi_params) for pi_params in self.config.policy_params
        ]

        policy_map: Dict[str, Policy] = {
            policy_params.name: PolicylCls(
                name=policy_params.name,
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                model=ModelCls,
                config=self.config,
                **policy_params.config
            )
            for policy_params in policy_params
        }
        experience_queue: Dict[str, ExperienceQueue] = {
            policy_name: ExperienceQueue(self.config)
            for policy_name in policy_map
        }
        while not self.stop_event.is_set():
            self.step(policy_map, experience_queue)

    def stop(self):
        self.stop_event.set()
        self.join()
    def push_batch(self, batch):
        self.experience_queue.put(batch)

    def get_metrics(self):
        metrics = []
        try:
            while not self.experience_queue.empty():
                metrics.append(self.metrics_queue.get(timeout=1e-3))
        except queue.Empty:
            pass

        return metrics

    def step(
            self,
            policy_map: Dict[str, Policy],
            experience_queue: Dict[str, ExperienceQueue]
    ):

        while not self.experience_queue.empty():
            experience_batch: SampleBatch = self.experience_queue.get()
            GlobalCounter.incr("trainer_batch_count")
            experience_queue[experience_batch.get_owner()].push([experience_batch])


        try:
            training_metrics = {}
            next_params = []
            for policy_name, policy_queue in experience_queue.items():
                if policy_queue.is_ready():
                    self.lock.acquire()
                    train_results = policy_map[policy_name].train(
                        policy_queue.pull(self.config.train_batch_size)
                    )
                    self.lock.release()
                    training_metrics[f"{policy_name}"] = train_results
                    GlobalCounter.incr(GlobalCounter.STEP)
                    next_params.append(policy_map[policy_name].get_params())

            # Push metrics to the metrics queue
            if len(training_metrics)> 0:
                self.metrics_queue.put(training_metrics)
            if len(next_params)>0:
                self.metrics_queue.put(next_params)

        except queue.Empty:
            pass







import importlib
import queue
import threading
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import tree
from ml_collections import ConfigDict

from polaris.checkpointing.checkpointable import Checkpointable
from polaris.experience.episode import EpisodeMetrics, NamedPolicyMetrics
from polaris.experience.worker_set import AsyncWorkerSet
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import Policy, PolicyParams, ParamsMap
from polaris.experience.matchmaking import RandomMatchmaking
from polaris.experience.sampling import ExperienceQueue, SampleBatch
from polaris.utils.metrics import MetricBank, GlobalCounter, GlobalTimer

import psutil


class AsyncTrainer(Checkpointable):
    def __init__(
            self,
            config: ConfigDict,
            restore=False,
    ):
        """
        TODO: Out-of-date
        """


        self.config = config
        self.worker_set = AsyncWorkerSet(
            config
        )

        # Init environment
        self.env = PolarisEnv.make(self.config.env, env_index=-1, **self.config.env_config)

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
                options=policy_param.options,
                # For any algo that needs to track either we have the online model
                is_online=True,
            )
            for policy_param in policy_params
        }

        self.params_map = ParamsMap(**{
            name: p.get_params() for name, p in self.policy_map.items()
        })

        self.experience_queue: Dict[str, ExperienceQueue] = {
            policy_name: ExperienceQueue(self.config)
            for policy_name in self.policy_map
        }

        self.matchmaking = RandomMatchmaking(
            agent_ids=self.env.get_agent_ids(),
        )

        self.runnning_jobs = []

        self.metricbank = MetricBank(
            dirname=self.config.tensorboard_logdir,
            report_dir=f"polaris_results/{self.config.seed}",
            report_freq=self.config.report_freq
        )
        self.metrics = self.metricbank.metrics

        # self.grad_thread = GradientThread(
        #     env=self.env,
        #     config=self.config
        # )
        # self.grad_lock = self.grad_thread.lock
        # self.grad_thread.start()

        super().__init__(
            checkpoint_config = config.checkpoint_config,

            components={
                "matchmaking": self.matchmaking,
                "config": self.config,
                "params_map": self.params_map,
                "metrics": self.metrics,
            }
        )

        if restore:
            if isinstance(restore, str):
                self.restore(restore_path=restore)
            else:
                self.restore()

            # Need to pass the restored references afterward
            self.metricbank.metrics = self.metrics

            env_step_counter = "counters/" + GlobalCounter.ENV_STEPS
            if env_step_counter in self.metrics:
                GlobalCounter[GlobalCounter.ENV_STEPS] = self.metrics["counters/" + GlobalCounter.ENV_STEPS].get()

            for policy_name, policy in self.policy_map.items():
                policy.setup(self.params_map[policy_name])

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
        jobs = [self.matchmaking.next(self.params_map) for _ in range(n_jobs)]
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
                exp_batch[SampleBatch.SEQ_LENS] = np.array(exp_batch[SampleBatch.SEQ_LENS])
                #if owner.is_recurrent:
                #    exp_batch = exp_batch.pad_and_split_to_sequences()
                self.experience_queue[owner.name].push([exp_batch])
                GlobalCounter.incr("batch_count")
                frames += exp_batch.size()
        if frames > 0:
            GlobalTimer[GlobalTimer.PREV_FRAMES] = time.time()
            prev_frames_dt = GlobalTimer.dt(GlobalTimer.PREV_FRAMES)
        if num_batch > 0:
            enqueue_time_ms = (time.time() - enqueue_time_start) * 1000.
        else:
            enqueue_time_ms = None

        GlobalCounter[GlobalCounter.ENV_STEPS] += env_steps
        GlobalCounter[GlobalCounter.NUM_EPISODES] += len(experience_metrics)

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

        # Make it policy specific, thus extract metrics of policies.

        if len(training_metrics)> 0:
            for policy_name, policy_training_metrics in training_metrics.items():
                policy_training_metrics = mean_metric_batch([policy_training_metrics])
                self.metricbank.update(policy_training_metrics, prefix=f"training/{policy_name}/",
                                       smoothing=self.config.training_metrics_smoothing)
        if len(experience_metrics) > 0:
            for metrics in experience_metrics:
                self.metricbank.update(tree.flatten_with_path(metrics), prefix=f"experience/",
                                       smoothing=self.config.episode_metrics_smoothing)

            # policy_experience_metrics = defaultdict(list)
            # pure_episode_metrics = []
            # for episode_metrics in experience_metrics:
            #     for policy_name, metrics in episode_metrics.policy_metrics.items():
            #         policy_experience_metrics[policy_name].append(metrics)
            #
            #     episode_metrics = episode_metrics._asdict()
            #     del episode_metrics["policy_metrics"]
            #     pure_episode_metrics.append(EpisodeMetrics(**episode_metrics))
            # mean_batched_experience_metrics = mean_metric_batch(pure_episode_metrics)
            # self.metricbank.update(mean_batched_experience_metrics, prefix="experience/",
            #                     smoothing=self.config.episode_metrics_smoothing)
            # for policy_name, metrics in policy_experience_metrics.items():
            #     metrics = mean_metric_batch(metrics)
            #     self.metricbank.update(metrics, prefix=f"experience/{policy_name}/", smoothing=self.config.episode_metrics_smoothing)

        misc_metrics =  [
                    ("RAM", psutil.virtual_memory().percent),
                    ("CPU", psutil.cpu_percent())
                ] + [
                    (f'{pi}_queue_length', queue.size())
                    for pi, queue in self.experience_queue.items()
                ]
        if frames > 0:
            misc_metrics.append(("FPS", frames / prev_frames_dt))
        if enqueue_time_ms is not None:
            misc_metrics.append(("experience_enqueue_ms", enqueue_time_ms))


        self.metricbank.update(
            misc_metrics
            , prefix="misc/", smoothing=0.9
        )

        self.metricbank.update(
            self.matchmaking.metrics(),
            prefix="matchmaking", smoothing=0.9
        )



        # We should call those only at the report freq...
        self.metricbank.update(
            tree.flatten_with_path(GlobalCounter.get()), prefix="counters/"
        )
        # self.metrics.update(
        #     tree.flatten_with_path(GlobalCounter), prefix="timers/", smoothing=0.9
        # )

        t.append(time.time())

        #print(np.diff(t))

        #print(GlobalCounter.dict)


    def run(self):
        try:
            while not self.is_done(self.metricbank.get()):
                self.training_step()
                self.metricbank.report(print_metrics=True)
                self.checkpoint_if_needed()
        except KeyboardInterrupt:
            print("Caught C^.")
            self.save()
        except Exception as e:
            print(e)
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









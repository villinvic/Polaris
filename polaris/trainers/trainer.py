import importlib
from typing import Dict

from ml_collections import ConfigDict
from polaris.experience.worker_set import WorkerSet
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import Policy, PolicyParams
from polaris.experience.matchmaking import RandomMatchmaking
from polaris.experience.sampling import ExperienceQueue
from polaris.utils.metrics import MetricBank, GlobalCounter


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
        PolicylCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        ModelCls = getattr(importlib.import_module(self.config.model_path), self.config.model_class)

        policy_params = [
            PolicyParams(*pi_params) for pi_params in self.config.policy_params
        ]
        self.policy_map: Dict[str, Policy] = {
            policy_params.name: PolicylCls(
                name=policy_params.name,
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                model = ModelCls,
                config=self.config,
                **policy_params.config
            )
            for policy_params in policy_params
        }

        self.experience_queue: Dict[str, ExperienceQueue] = {
            policy_name: ExperienceQueue(self.config)
            for policy_name in self.policy_map
        }

        self.matchmaking = RandomMatchmaking(
            agent_ids=self.env.get_agent_ids(),
            policies=list(self.policy_map.values())
        )

        self.runnning_jobs = []

        self.metrics = MetricBank(
            dirname=self.config.tensorboard_logdir,
            report_freq=self.config.report_freq
        )

    def training_step(self):
        """
        Executes one iteration of the trainer.
        :return: Training iteration results
        """

        n_jobs = self.worker_set.get_num_worker_available()
        jobs = [self.matchmaking.next() for _ in range(n_jobs)]
        self.runnning_jobs += self.worker_set.push_jobs(jobs)

        ready = {
            policy_name for policy_name, queue in self.experience_queue.items()
            if queue.is_ready()
        }

        if len(ready) == 0:
            sample_batches = self.worker_set.wait(self.runnning_jobs)
            for b in sample_batches:
                self.experience_queue[b.get_owner()].push(b)

        else:
            for policy_name in ready:
                train_results = self.policy_map[policy_name].train(
                    self.experience_queue[policy_name].pull(self.config.train_batch_size)
                )
                self.metrics.update(train_results)
            GlobalCounter["step"] += 1

    def run(self):
        try:
            iteration = 0
            while True:
                self.training_step()
                self.metrics.report()
                iteration += 1
                print("Iteration:", iteration)
        except KeyboardInterrupt:
            print("Caught C^.")








from typing import Dict, Generator

import numpy as np
from ml_collections import ConfigDict
import ray
from polaris.experience.sampling import SampleBatch
from polaris.experience.episode import Episode
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import PolicyParams, Policy
from polaris.policies.random import RandomPolicy

import importlib

from polaris.environments.example import DummyEnv, PolarisCartPole


@ray.remote(num_cpus=1, num_gpus=0)
class EnvWorker:

    def __init__(
            self,
            *,
            worker_id: int,
            config: ConfigDict,
    ):
        self.worker_id = worker_id
        self.config = config

        # Init environment
        self.env = PolarisEnv.make(self.config.env)

        # We can extend to multiple policy types if really needed, but won't be memory efficient
        PolicyCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        ModelCls = getattr(importlib.import_module(self.config.model_path), self.config.model_class)

        self.policy_placeholders: Dict[str, Policy] = {
            f"parametrised_{aid}": PolicyCls(
                name="placeholder",
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                model=ModelCls,
                config=self.config
            )
            for aid in self.env.get_agent_ids()
        }
        random_policy = RandomPolicy(self.env.action_space, self.config)
        for aid in self.env.get_agent_ids():
            self.policy_placeholders[f"random_{aid}"] = random_policy

        self.sample_buffer = {
            aid: SampleBatch(self.config.batch_size) for aid in self.env.get_agent_ids()
        }

    def run_episode_for(self, agent_ids_to_policy_params: Dict[str, PolicyParams], episode_options=None) -> Generator:

        # Init environment
        if self.env is None:
            self.env = PolarisEnv.make(self.config.env)

        agents_to_policies = {
            aid: self.policy_placeholders[policy_params.policy_type + f"_{aid}"].setup(policy_params)
            for aid, policy_params in agent_ids_to_policy_params.items()
        }

        try:
            for batches in Episode(
                    self.env,
                    agents_to_policies,
                    self.config
            ).run(
                self.sample_buffer,
                options=episode_options
            ):
                yield self.worker_id, batches

        except Exception as e:
            print("TODO: restart env if failed")
            print(e)
            raise e
            self.env.close()
            # TODO : recall run_episode_for

        # Episode finished
        # TODO : send metrics through here
        yield self.worker_id, None

if __name__ == '__main__':
    env = DummyEnv()
    env.register()

    config = ConfigDict()
    config.env = env.env_id
    config.policy_model_path = "polaris.policies.policy"
    config.policy_model_class = "DummyPolicy"
    config.policy_config = {}
    config.batch_size = 8
    config.lock()

    workers = [EnvWorker.remote(worker_id=wid, config=config) for wid in range(1)]

    policy_params = [
        PolicyParams(name="bobby", policy_type="parametrised"),
        PolicyParams(name="alexi", policy_type="parametrised"),
        PolicyParams(name="randi", policy_type="random"),
    ]
    jobs = []

    for worker in workers:
        jobs.append(worker.run_episode_for.remote(
            {
                aid: policy_params[np.random.choice(len(policy_params))] for aid in env.get_agent_ids()
            }
        ))

    sample_queue = []

    while len(jobs) > 0:

        done_jobs, _ = ray.wait(
            jobs,
            # num_returns=1,
            # timeout=None,
            # fetch_local=False,
        )
        for job in done_jobs:
            wid, sample_batch = ray.get(next(job))
            if sample_batch is None:
                jobs.remove(job)
                print("we are done")
            else:
                sample_queue.extend(sample_batch)

    for batch in sample_queue:
        print(batch[SampleBatch.POLICY_ID])


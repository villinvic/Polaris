from typing import Dict, Generator
import time
import numpy as np
from ml_collections import ConfigDict
import ray

from polaris.policies import Policy, PolicyParams, RandomPolicy
from polaris.experience.sampling import SampleBatch
from polaris.experience.episode import Episode
from polaris.environments.polaris_env import PolarisEnv



import importlib


@ray.remote(num_cpus=1, num_gpus=0)
class EnvWorker:

    def __init__(
            self,
            *,
            worker_id: int,
            config: ConfigDict,
    ):
        import tensorflow as tf
        tf.compat.v1.enable_eager_execution()

        self.worker_id = worker_id
        self.config = config

        # Init environment
        self.env = PolarisEnv.make(self.config.env, env_index=self.worker_id, **self.config.env_config)

        # We can extend to multiple policy types if really needed, but won't be memory efficient
        PolicyCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)
        self.policy_placeholders: Dict[str, Policy] = {
            f"parametrised_{aid}": PolicyCls(
                name="placeholder",
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                config=self.config,
                options={},
                policy_config=self.config.default_policy_config
            )
            for aid in self.env.get_agent_ids()
        }
        random_policy = RandomPolicy(self.env.action_space, self.config)
        for aid in self.env.get_agent_ids():
            self.policy_placeholders[f"random_{aid}"] = random_policy

        self.sample_buffer = {
            aid: SampleBatch(self.config.trajectory_length, max_seq_len=self.config.max_seq_len) for aid in self.env.get_agent_ids()
        }

    def run_episode_for(self, agent_ids_to_policy_params: Dict[str, PolicyParams]) -> Generator:
        # Init environment
        if self.env is None:
            self.env = PolarisEnv.make(self.config.env, env_index=self.worker_id, **self.config.env_config)

        agents_to_policies = {
            aid: self.policy_placeholders[policy_params.policy_type + f"_{aid}"].setup(policy_params)
            for aid, policy_params in agent_ids_to_policy_params.items()
        }
        episode = Episode(
            self.env,
            agents_to_policies,
            self.config
        )
        try:
            for batches in episode.run(
                self.sample_buffer,
            ):
                yield self.worker_id, batches

        except Exception as e:
            print("Restarting environment:", e)
            self.env.close()
            time.sleep(2)
            self.env = PolarisEnv.make(self.config.env, env_index=self.worker_id, **self.config.env_config)
            # TODO : recall run_episode_for

        # Episode finished
        yield self.worker_id, episode.metrics

if __name__ == '__main__':
    from polaris.environments.example import DummyEnv

    env = DummyEnv()
    env.register()

    config = ConfigDict()
    config.env = env.env_id
    config.policy_model_path = "polaris.policies.policy"
    config.policy_model_class = "DummyPolicy"
    config.policy_config = {}
    config.trajectory_length = 8
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


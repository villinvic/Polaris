from typing import Dict, Generator
import time
import numpy as np
from ml_collections import ConfigDict
import ray
import copy
from gymnasium.error import ResetNeeded

from polaris.policies import Policy, PolicyParams, RandomPolicy
from polaris.experience.sampling import SampleBatch
from polaris.experience.episode import Episode, EpisodeSpectator
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.utils.gae import compute_gae_for_sample_batch
import importlib


@ray.remote(num_cpus=1, num_gpus=0)
class EnvWorker:

    def __init__(
            self,
            *,
            worker_id: int,
            config: ConfigDict,
    ):

        self.initialised = False
        self.callbacks = config.episode_callback_class(config)

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
                stats={},
                policy_config=copy.deepcopy(self.config.default_policy_config)
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
        if not self.initialised:
            self.initialised = True

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

        except ResetNeeded as e:
            print("Restarting environment:", e)
            self.env.close()
            time.sleep(2)
            self.env = PolarisEnv.make(self.config.env, env_index=self.worker_id, **self.config.env_config)
            # TODO : recall run_episode_for


        # Episode finished
        yield self.worker_id, [episode.metrics]

        del episode



@ray.remote(num_gpus=0, num_cpus=1)
class SyncEnvWorker:

    def __init__(
            self,
            *,
            worker_id: int,
            config: ConfigDict,
            spectator=False,
    ):
        self.initialised = False
        self.spectator = spectator
        self.callbacks = config.episode_callback_class(config)

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
                stats={},
                policy_config=copy.deepcopy(self.config.default_policy_config)
            )
            for aid in self.env.get_agent_ids()
        }
        random_policy = RandomPolicy(self.env.action_space, self.config)
        for aid in self.env.get_agent_ids():
            self.policy_placeholders[f"random_{aid}"] = random_policy

        self.sample_buffer = {
            aid: SampleBatch(self.config.trajectory_length, max_seq_len=self.config.max_seq_len) for aid in self.env.get_agent_ids()
        }

        self.current_episode = None
        self.env_runner = None

    def get_next_batch_for(self, agent_ids_to_policy_params: Dict[str, PolicyParams]):
        # Init environment
        if self.env is None:
            self.env = PolarisEnv.make(self.config.env, env_index=self.worker_id, **self.config.env_config)
        if not self.initialised:
            self.initialised = True

        agents_to_policies = {
            aid: self.policy_placeholders[policy_params.policy_type + f"_{aid}"].setup(policy_params)
            for aid, policy_params in agent_ids_to_policy_params.items()
        }

        try:
            if self.current_episode is None:
                if self.spectator:
                    self.current_episode = EpisodeSpectator(
                        self.env,
                        agents_to_policies,
                        self.callbacks,
                        self.config
                    )
                else:
                    self.current_episode = Episode(
                        self.env,
                        agents_to_policies,
                        self.callbacks,
                        self.config
                    )
                self.env_runner = self.current_episode.run(self.sample_buffer)

            batches = next(self.env_runner)
            done = False
            if self.config.compute_advantages_on_workers:
                for batch in batches:
                    aid = batch.get_aid()
                    compute_gae_for_sample_batch(policy=agents_to_policies[aid],
                                                 sample_batch=batch
                                                 )
                    done = batch[SampleBatch.DONE][-1]

            if done:
                try:
                    next(self.env_runner)
                except StopIteration:
                    # Episode finished
                    self.env_runner = None
                    metrics = self.current_episode.metrics
                    self.current_episode = None
                    batches += [metrics]

            return self.worker_id, batches

        except RuntimeError as e:
            # Episode finished for spectator
            self.env_runner = None
            metrics = self.current_episode.metrics
            self.current_episode = None
            return self.worker_id, [metrics]

        except ResetNeeded as e:
            print("Restarting environment:", e)
            self.env_runner = None
            self.current_episode = None

            self.env.close()
            time.sleep(2)
            self.env = PolarisEnv.make(self.config.env, env_index=self.worker_id, **self.config.env_config)
            return self.worker_id, [None]
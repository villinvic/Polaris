from typing import Dict, Generator, List, Tuple, Any
import time

import tree
from ml_collections import ConfigDict
import ray
import copy
from gymnasium.error import ResetNeeded

from polaris.experience.vectorised_episode import VectorisableEpisode, EpisodeState
from polaris.experience.timestep import TimeStep
from polaris.policies import Policy, PolicyParams, RandomPolicy
from polaris.experience.sampling import SampleBatch
from polaris.experience.episode import Episode, EpisodeSpectator, EpisodeMetrics
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



def map_index(batched, i):
    return tree.map_structure(
        lambda v: v[i],
        batched
    )


@ray.remote(num_gpus=0, num_cpus=1)
class VectorisedEnvWorker:

    def __init__(
            self,
            *,
            worker_id: int,
            config: ConfigDict,
    ):

        self.config = config
        self.callbacks = config.episode_callback_class(config)
        self.num_envs = config.num_envs_per_worker
        self.worker_id = worker_id
        self.config = config

        self.envs = [
                PolarisEnv.make(self.config.env, env_index=self.worker_id * self.num_envs + env_id, **self.config.env_config)
                for env_id in range(self.num_envs)
            ]

        self.episodes = [
            VectorisableEpisode(
                env,
                self.callbacks,
                config
            )
            for env in self.envs
        ]

    def step(
            self,
            actions,
            options,
    ) -> Tuple[List[Dict[Any, TimeStep]], List[EpisodeMetrics], int]:
        timesteps = []
        episode_metrics = []

        for i, episode in enumerate(self.episodes):
            try:
                # TODO: only works with discrete action spaces
                timesteps.append(
                    episode.step(map_index(actions, i),
                                 options,
                                 #map_index(options, i)
                                 ))
            except ResetNeeded as e:
                print(f"Restarting environment {episode.env.env_index}:", e)
                episode.env.close()
                time.sleep(1)
                episode.state = EpisodeState.CRASHED

            if episode.state in (EpisodeState.TERMINATED, EpisodeState.CRASHED):
                metrics = episode.get_metrics()
                if metrics is not None:
                    episode_metrics.append(episode.get_metrics())

                episode.reset()

        return timesteps, episode_metrics, self.worker_id




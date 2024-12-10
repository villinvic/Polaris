import time
import copy
from collections import defaultdict
from typing import Iterator, List, NamedTuple, Dict

import numpy as np
import tree
from ml_collections import ConfigDict
from gymnasium.error import ResetNeeded

from .sampling import SampleBatch

class NamedPolicyMetrics(NamedTuple):
    agent: str = None
    policy: str = None
    returns: np.float32 = 0.
    episode_length: np.int32 = 0
    custom_metrics: dict = {}

    @property
    def metrics(self):
        return PolicyMetrics(returns=self.returns, episode_length=self.episode_length, custom_metrics=self.custom_metrics)

class PolicyMetrics(NamedTuple):
    returns: np.float32 = 0.
    episode_length: np.int32 = 0
    custom_metrics: dict = {}

class EpisodeMetrics(NamedTuple):
    total_returns: np.float32 = 0.
    length: np.int32 = 0
    stepping_time_ms: np.float32 = 100.
    matchmaking_wait_ms: np.float32 = 100.
    custom_metrics: dict = {}
    policy_metrics: Dict[str, PolicyMetrics] = {}


class Episode:

    def __init__(
            self,
            environment,
            agents_to_policies,
            callbacks,
            config: ConfigDict,
    ):
        self.env = environment
        self.agents_to_policies = agents_to_policies
        self.callbacks = callbacks
        self.config = config
        self.id = np.random.randint(1e9)
        self.metrics = EpisodeMetrics()
        self.custom_metrics = {}
        self.last_episode_end = time.time()

    def run(
            self,
            sample_batches,
    ) -> Iterator[List[SampleBatch]]:
        """
        :param sample_batches: batches to fill for each policy, can come from a previous experience. Pushes to the trainer proc every times a batch has been filled
        :return: partially filled sample_batches
        # TODO: we could pass some additional episode options if needed
        """
        t1 = time.time()

        states = {
            aid: policy.get_initial_state() for aid, policy in self.agents_to_policies.items()
        }
        next_states = {}
        actions = {}
        extras = {}

        prev_actions = {aid: 0 for aid in self.agents_to_policies}
        prev_rewards = {aid: 0. for aid in self.agents_to_policies}
        dones = {
            aid: False for aid in self.agents_to_policies
        }
        dones["__all__"] = False
        observations, infos = self.env.reset(options={
            aid: p.options for aid, p in self.agents_to_policies.items()
        })

        t = 0
        episode_lengths = defaultdict(np.int32)
        episode_rewards = defaultdict(np.float32)

        while not dones["__all__"]:
            for aid, policy in self.agents_to_policies.items():
                actions[aid], next_states[aid], extras[aid] = policy.compute_single_action_with_extras(
                    obs=observations[aid],
                    prev_action=prev_actions[aid],
                    prev_reward=prev_rewards[aid],
                    state=states[aid]
                )

            try:
                next_observations, rewards, dones, truncs, infos = self.env.step(actions)
            except ResetNeeded as e:
                for aid, sample_batch in sample_batches.items():
                    sample_batch.reset()
                raise e

            dones["__all__"] = dones["__all__"] or truncs["__all__"]


            self.callbacks.on_step(
                self.agents_to_policies,
                actions,
                observations,
                next_observations,
                rewards,
                dones,
                infos,
                self.custom_metrics
            )

            batches = []

            for aid, policy in self.agents_to_policies.items():
                if not dones[aid] or dones["__all__"]:
                    # TODO: check on done
                    done = dones[aid] or dones["__all__"]
                    batches += sample_batches[aid].push(
                        {
                            SampleBatch.OBS: observations[aid],
                            SampleBatch.NEXT_OBS: next_observations[aid],
                            SampleBatch.PREV_ACTION: prev_actions[aid],
                            SampleBatch.ACTION: actions[aid],

                            SampleBatch.REWARD: rewards[aid],
                            SampleBatch.PREV_REWARD: prev_rewards[aid],
                            SampleBatch.DONE: done,
                            SampleBatch.STATE: states[aid],
                            SampleBatch.NEXT_STATE: next_states[aid],
                            SampleBatch.AGENT_ID: aid,
                            SampleBatch.POLICY_ID: policy.name,
                            SampleBatch.VERSION: policy.version,
                            SampleBatch.EPISODE_ID: self.id,
                            SampleBatch.T: t,

                            # SampleBatch.VALUES: values[aid],
                            # SampleBatch.ACTION_LOGP: action_logp[aid],
                            # SampleBatch.ACTION_LOGITS: action_logits[aid],

                            **extras[aid],
                        },
                        flush=done
                    )

                    episode_lengths[aid] += 1
                    episode_rewards[aid] += rewards[aid]

                    observations[aid] = copy.deepcopy(next_observations[aid])
                    prev_rewards[aid] = rewards[aid]
                    prev_actions[aid] = actions[aid]
                    states[aid] = copy.deepcopy(next_states[aid])

            if len(batches) > 0:
                self.callbacks.on_trajectory_end(
                    self.agents_to_policies,
                    batches,
                    self.custom_metrics
                )

                yield batches
            t += 1


        self.callbacks.on_episode_end(
            self.agents_to_policies,
            self.env.get_episode_metrics(),
            self.custom_metrics
        )

        self.metrics = EpisodeMetrics(
            stepping_time_ms=1000.*np.float32(time.time()-t1) / t,
            matchmaking_wait_ms=1000.*np.float32(t1-self.last_episode_end),
            total_returns=sum(episode_rewards.values()),
            custom_metrics=self.custom_metrics,
            length=np.int32(t),
            policy_metrics={policy.name: PolicyMetrics(
            returns=episode_rewards[agent_id],
            episode_length=episode_lengths[agent_id])
                for agent_id, policy in self.agents_to_policies.items()}
        )
        self.last_episode_end = time.time()
        # pass the batch for next episodes


class EpisodeSpectator(Episode):

    def run(
            self,
            sample_batches,
    ) -> Iterator[List[SampleBatch]]:
        """
        :param sample_batches: batches to fill for each policy, can come from a previous experience. Pushes to the trainer proc every times a batch has been filled
        :return: partially filled sample_batches
        """

        t1 = time.time()

        states = {
            aid: policy.get_initial_state() for aid, policy in self.agents_to_policies.items()
        }

        actions = {aid: 0 for aid in self.agents_to_policies}
        rewards = {aid: 0. for aid in self.agents_to_policies}
        dones = {
            aid: False for aid in self.agents_to_policies
        }
        dones["__all__"] = False
        observations, infos = self.env.reset(options={
            aid: p.options for aid, p in self.agents_to_policies.items()
        })

        t = 0
        episode_lengths = defaultdict(np.int32)
        episode_rewards = defaultdict(np.float32)

        while not dones["__all__"]:
            for aid, policy in self.agents_to_policies.items():

                actions[aid], states[aid] = policy.compute_single_action(
                    obs=observations[aid],
                    prev_action=actions[aid],
                    prev_reward=rewards[aid],
                    state=states[aid]
                )

            try:
                observations, rewards, dones, truncs, infos = self.env.step(actions)
            except ResetNeeded as e:
                for aid, sample_batch in sample_batches.items():
                    sample_batch.reset()
                raise e

            dones["__all__"] = dones["__all__"] or truncs["__all__"]


        # TODO: cleanup
        self.metrics = EpisodeMetrics(
            stepping_time_ms=1000.*np.float32(time.time()-t1) / t,
            matchmaking_wait_ms=1000.*np.float32(t1-self.last_episode_end),
            total_returns=sum(episode_rewards.values()),
            custom_metrics=self.custom_metrics,
            length=np.int32(t),
            policy_metrics={policy.name: PolicyMetrics(
            returns=episode_rewards[agent_id],
            episode_length=episode_lengths[agent_id])
                for agent_id, policy in self.agents_to_policies.items()}
        )
        self.last_episode_end = time.time()
        # pass the batch for next episodes
        
        raise StopIteration
        yield
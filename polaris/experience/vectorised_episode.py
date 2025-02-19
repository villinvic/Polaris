from collections import defaultdict
from enum import Enum
from typing import Dict, Any
import numpy as np
from ml_collections import ConfigDict

from polaris.environments import PolarisEnv
from polaris.experience.episode import EpisodeMetrics
from polaris.experience.timestep import TimeStep


class EpisodeState(Enum):
    UNINITIALISED = 0
    RESET = 1
    RUNNING = 2
    TERMINATED = 3
    CRASHED = 4


class VectorisableEpisode:
    def __init__(
            self,
            env: PolarisEnv,
            callbacks,
            config: ConfigDict,
    ):
        self.env = env
        self.id = np.random.randint(1e9)
        self.metrics = None
        self.callbacks = callbacks
        self.config = config
        self.state = EpisodeState.UNINITIALISED
        self.t = 0

        self.episode_lengths = defaultdict(np.int32)
        self.episode_rewards = defaultdict(np.float32)

    def reset(self):
        self.__init__(self.env, self.callbacks, self.config)

    def get_metrics(self):
        return self.metrics

    def step(
            self,
            actions: Dict,
            options: Dict
    )-> Dict[Any, TimeStep]:
        # TODO: handle errors and resets
        # TODO: callbacks

        if self.state == EpisodeState.UNINITIALISED:

            observations, _ = self.env.reset(options=options)
            self.state = EpisodeState.RESET

            return {
                aid: TimeStep(
                    obs=observations[aid],
                    agent_id=aid,
                    episode_id=self.id,
                    env_index=self.env.env_index,
                    episode_state=self.state,
                    t=self.t
                )
                for aid in self.env.get_agent_ids()
            }

        self.state = EpisodeState.RUNNING

        observations, rewards, dones, truncateds, _ = self.env.step(actions)

        self.t += 1

        done = dones["__all__"] or truncateds["__all__"]

        for aid in self.env.get_agent_ids():
            self.episode_rewards[aid] += rewards[aid]
            if not dones[aid]:
                self.episode_lengths[aid] += 1

        if done:
            # TODO: episodic per policy metrics (we do not have access to policy ids here)
            self.state = EpisodeState.TERMINATED
            self.metrics = EpisodeMetrics(
                stepping_time_ms=0,
                matchmaking_wait_ms=0,
                total_returns=sum(self.episode_rewards.values()),
                custom_metrics=self.env.get_episode_metrics(),
                length=np.int32(self.t),
                policy_metrics={}
            )


        return {
                aid: TimeStep(
                    obs=observations[aid],
                    agent_id=aid,
                    episode_id=self.id,
                    env_index=self.env.env_index,
                    t=self.t,
                    episode_state=self.state,
                    reward=rewards[aid],
                    done=done # or dones[aid]
                )
                for aid in self.env.get_agent_ids()
            }
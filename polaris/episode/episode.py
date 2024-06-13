from typing import Iterator

import numpy as np
from ml_collections import ConfigDict

from polaris.episode import SampleBatch


class Episode:


    def __init__(
            self,
            environment,
            agents_to_policies,
            config: ConfigDict,
    ):
        self.env = environment
        self.agents_to_policies = agents_to_policies
        self.config = config
        self.id = np.random.randint(1e9)

        self.metrics = {}


    def run(
            self,
            sample_batches,
            options=None
    ) -> Iterator[SampleBatch]:
        """
        :param sample_batches: batches to fill for each policy, can come from a previous episode. Pushes to the trainer proc every times a batch has been filled
        :param options: passed to the env.reset() method
        :return: partially filled sample_batches
        """

        states = {
            aid: policy.get_initial_state() for aid, policy in self.agents_to_policies.items()
        }
        next_states = {}
        actions = {}
        prev_actions = {0 for _ in self.agents_to_policies}
        prev_rewards = {0 for _ in self.agents_to_policies}
        dones = {
            aid: False for aid in self.agents_to_policies
        }
        dones["__all__"] = False
        observations, infos = self.env.reset(options)

        while not dones["__all__"]:
            for aid, policy in self.agents_to_policies.items():
                actions[aid], next_states[aid] = policy.compute_action(
                    observations[aid],
                    prev_action=prev_actions[aid],
                    prev_reward=prev_rewards[aid],
                    states=states[aid]
                )

            next_observations, rewards, dones, _, infos = self.env.step(actions)
            # self.config.callbacks.on_env_step(
            #     actions,
            #     observations,
            #     next_observations,
            #     rewards,
            #     dones,
            #     infos,
            #     self.agents_to_policies,
            #     self.metrics
            # )


            batches = []
            for aid, policy in self.agents_to_policies.items():
                if not dones[aid] or dones["__all__"]:
                    batches += sample_batches[aid].push(
                        {
                        SampleBatch.OBS: observations[aid],
                        SampleBatch.ACTION: actions[aid],
                        SampleBatch.REWARD: rewards[aid],
                        SampleBatch.DONE: dones[aid],
                        SampleBatch.AGENT_ID: aid,
                        SampleBatch.POLICY_ID: policy.name,
                        SampleBatch.EPISODE_ID: self.id
                        }
                    )
            if len(batches) > 0:
                yield batches
            observations = next_observations
            prev_rewards = rewards
            prev_actions = actions

        # pass the batch for next episodes
        return sample_batches





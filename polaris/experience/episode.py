from typing import Iterator, List

import numpy as np
from ml_collections import ConfigDict

from .sampling import SampleBatch


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
    ) -> Iterator[List[SampleBatch]]:
        """
        :param sample_batches: batches to fill for each policy, can come from a previous experience. Pushes to the trainer proc every times a batch has been filled
        :param options: passed to the env.reset() method
        :return: partially filled sample_batches
        """

        states = {
            aid: policy.get_initial_state() for aid, policy in self.agents_to_policies.items()
        }
        next_states = {}
        actions = {}
        action_logp = {}
        action_logits = {}

        prev_actions = {aid: 0 for aid in self.agents_to_policies}
        prev_rewards = {aid: 0 for aid in self.agents_to_policies}
        dones = {
            aid: False for aid in self.agents_to_policies
        }
        dones["__all__"] = False
        observations, infos = self.env.reset(options=options)

        t = 0

        while not dones["__all__"]:
            for aid, policy in self.agents_to_policies.items():
                actions[aid], next_states[aid], action_logp[aid], action_logits[aid] = policy.compute_action(
                    {
                        SampleBatch.OBS: observations[aid],
                        SampleBatch.PREV_ACTION: prev_actions[aid],
                        SampleBatch.PREV_REWARD: prev_rewards[aid],
                        SampleBatch.STATE: states[aid]
                    }
                )

            next_observations, rewards, dones, truncs, infos = self.env.step(actions)
            dones["__all__"] = dones["__all__"] or truncs["__all__"]
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
                            SampleBatch.NEXT_OBS: next_observations[aid],
                            SampleBatch.PREV_ACTION: prev_actions[aid],
                            SampleBatch.ACTION: actions[aid],
                            SampleBatch.ACTION_LOGP: action_logp[aid],
                            SampleBatch.ACTION_LOGITS: action_logits[aid],
                            SampleBatch.REWARD: rewards[aid],
                            SampleBatch.DONE: dones[aid],
                            SampleBatch.STATE: states[aid],
                            SampleBatch.NEXT_STATE: states[aid],
                            SampleBatch.AGENT_ID: aid,
                            SampleBatch.POLICY_ID: policy.name,
                            SampleBatch.VERSION: policy.version,
                            SampleBatch.EPISODE_ID: self.id,

                            SampleBatch.T: t,
                        }
                    )
            if len(batches) > 0:
                yield batches
            observations = next_observations
            prev_rewards = rewards
            prev_actions = actions
            t += 1

        # pass the batch for next episodes

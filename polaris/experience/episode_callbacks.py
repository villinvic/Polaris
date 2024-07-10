from typing import Dict, List


class EpisodeCallbacks:

    def on_step(
            self,
            agents_to_policies: Dict[str, "Policy"],
            actions: Dict,
            observations: Dict,
            next_observations: Dict,
            rewards: Dict,
            dones: Dict,
            infos: Dict,
            metrics: Dict,
    ):
        """
        :param metrics: should be edited to report custom metrics
        """
        pass

    def on_trajectory_end(
            self,
            agents_to_policies: Dict[str, "Policy"],
            sample_batches: List["SampleBatch"],
            metrics: Dict
    ):
        """
        :param metrics: should be edited to report custom metrics
        """
        pass

    def on_episode_end(
        self,
        agents_to_policies: Dict[str, "Policy"],
        env_metrics: Dict,
        metrics: Dict,
    ):
        pass
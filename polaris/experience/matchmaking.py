from typing import Dict, List, Union, Tuple, Any
from abc import ABC

import numpy as np

from polaris.policies.policy import Policy, PolicyParams


class MatchMaking(ABC):

    def __init__(
            self,
            agent_ids,
            policy_params: Dict[str, PolicyParams],
            **kwargs
    ):
        self.agent_ids = agent_ids
        self.policy_params = policy_params

    def next(
            self,
            **kwargs,
    ) -> Tuple[Dict[str, PolicyParams], Union[Dict, None]]:
        pass

class RandomMatchmaking(MatchMaking):

    def next(
            self,
            **kwargs,
    ) -> Tuple[Dict[str, PolicyParams], Any]:

        return {
            aid: list(self.policy_params.values())[np.random.choice(len(self.policy_params))] for aid in self.agent_ids
        }, None
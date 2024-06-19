from typing import Dict, List, Union, Tuple
from abc import ABC

import numpy as np

from polaris.policies.policy import Policy, PolicyParams


class MatchMaking(ABC):

    def __init__(
            self,
            agent_ids,
            policies: List[Policy],
            **kwargs
    ):
        self.agent_ids = agent_ids
        self.policies = policies

    def next(
            self,
            **kwargs,
    ) -> Tuple[Dict[str, PolicyParams], Union[Dict, None]]:
        pass

class RandomMatchmaking(MatchMaking):

    def next(
            self,
            **kwargs,
    ) -> Dict[str, Policy]:

        return {
            aid: self.policies[np.random.choice(len(self.policies))].get_params() for aid in self.agent_ids
        }, None
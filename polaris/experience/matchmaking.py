from collections import defaultdict
from typing import Dict, List, Union, Tuple, Any
from abc import ABC

import numpy as np


class MatchMaking(ABC):

    def __init__(
            self,
            agent_ids,
            **kwargs
    ):
        self.agent_ids = agent_ids

    def next(
            self,
            params_map: Dict[str, "PolicyParams"],
            **kwargs,
    ) -> Dict[str, "PolicyParams"]:
        pass

    def update(self, **kwargs):
        pass

    def metrics(self):
        return {}

class RandomMatchmaking(MatchMaking):

    def next(
            self,
            params_map: Dict[str, "PolicyParams"],
            **kwargs,
    ) -> Dict[str, "PolicyParams"]:

        return {
            aid: list(params_map.values())[np.random.choice(len(params_map), replace=len(params_map)<len(self.agent_ids))] for aid in self.agent_ids
        }


class Spectator(MatchMaking):
    pass


class TwoPlayerEloRanking(MatchMaking):

    def __init__(
            self,
            agent_ids,
            policy_params: Dict[str, "PolicyParams"],
            initial_elo=1000,
            initial_lr=40,
            annealing=0.99,
            final_lr=20,
            win_rate_lr=2e-2,
    ):
        
        super().__init__(agent_ids=agent_ids, policy_params=policy_params)

        self.initial_elo = initial_elo
        self.initial_lr = initial_lr
        self.annealing = annealing
        self.final_lr = final_lr

        self.match_count = defaultdict(int)
        self.elo_scores = defaultdict(lambda: self.initial_elo)
        self.lr = defaultdict(lambda: self.initial_lr)

        # We keep track for some more stats
        self.win_rate_lr = win_rate_lr
        self.win_rates = defaultdict(lambda: 0.5)


    def next(
            self,
            params_map: Dict[str, "PolicyParams"],
            **kwargs,
    ) -> Dict[str, "PolicyParams"]:
        # Do not make players play against themselves
        policy_params = list(params_map.values())
        return {
            aid: params for aid, params in
            zip(
                self.agent_ids,
                policy_params[np.random.choice(len(params_map), len(self.agent_ids), replace=False)]
            )
        }, None

    def expected_outcome(self, delta_elo):

        # 400 is just a score used for human normalisation
        return 1 / (1 + np.power(10, -delta_elo / 400.))

    def update(
            self,
            pid1: str,
            pid2: str,
            outcome: float
    ):



        delta_elo = self.elo_scores[pid1] - self.elo_scores[pid2]

        win_prob = self.expected_outcome(delta_elo)
        update = outcome - win_prob

        print(outcome)

        self.elo_scores[pid1] = self.elo_scores[pid1] + self.lr[pid1] * update
        self.elo_scores[pid2] = self.elo_scores[pid2] + self.lr[pid2] * (-update)

        self.match_count[pid1] += 1
        self.match_count[pid2] += 1

        self.lr[pid1] = np.maximum(self.final_lr, self.annealing * self.lr[pid1])
        self.lr[pid2] = np.maximum(self.final_lr, self.annealing * self.lr[pid2])

        self.win_rates[pid1] = self.win_rates[pid1] * (1 - self.win_rate_lr) + (1.-outcome) * self.win_rate_lr
        self.win_rates[pid2] = self.win_rates[pid2] * (1 - self.win_rate_lr) + outcome * self.win_rate_lr

    def metrics(self):
        return {
            "elos": dict(self.elo_scores),
            "match_count": dict(self.match_count),
            "win_rates": dict(self.win_rates),
            "elo_lrs": dict(self.lr)
        }



if __name__ == '__main__':

    r = TwoPlayerEloRanking(
        agent_ids=[1, 2],
        policy_params={
            "bob": PolicyParams(),
            "jack": PolicyParams()
        },
    )

    actual_wr = 0.57

    for t in range(20000):

        outcome = float(np.random.random() < actual_wr)
        r.update("bob", "jack", outcome=outcome)
        print(r.expected_outcome(r.elo_scores["bob"] - r.elo_scores["jack"]))

        print(r.metrics())


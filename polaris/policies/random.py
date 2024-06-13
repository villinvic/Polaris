from .policy import Policy

class RandomPolicy(Policy):

    def __init__(
            self,
            action_space,

    ):
        super().__init__(
            name="RandomPolicy",
            action_space=action_space,
            observation_space=None,
        )

    def compute_action(
            self,
            observation,
            states=None,
            prev_action=None,
            prev_reward=None

    ):
        return self.action_space.sample(), None
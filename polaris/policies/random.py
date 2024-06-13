from .policy import Policy

class RandomPolicy(Policy):

    def compute_action(
            self,
            observation,
            state=None,
            prev_action=None,
            prev_reward=None

    ):
        return self.action_space.sample()
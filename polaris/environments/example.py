from polaris.policies.random import RandomPolicy
from .polaris_env import PolarisEnv
from ray.tune.registry import _Registry, ENV_CREATOR
from polaris.episode import Episode, SampleBatch

from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class PolarisCartPole(CartPoleEnv, PolarisEnv):

    def __init__(self):

        PolarisEnv.__init__(self, env_id="cartpole")
        CartPoleEnv.__init__(self)
        self._agent_ids = {0}



if __name__ == '__main__':

    env = PolarisCartPole()
    env.register()
    reg = _Registry()

    retrieved_env = reg.get(ENV_CREATOR, "cartpole")()

    policies = [
        RandomPolicy(
            name="randompi1",
            action_space=retrieved_env.action_space,
            observation_space=retrieved_env.observation_space
            )
    ]
    agents_to_policies = {
        aid: pi for aid, pi in zip(retrieved_env.get_agent_ids(), policies)
    }

    sample_batches = {
        aid: SampleBatch(32) for aid in retrieved_env.get_agent_ids()
    }

    for batch in Episode(retrieved_env, agents_to_policies, None).run(sample_batches, policies):
        if batch.is_full():
            print(batch)




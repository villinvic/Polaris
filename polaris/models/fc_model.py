import sonnet as snt
from gymnasium.spaces import Discrete
import tensorflow as tf
from .base import BaseModel
from tensorflow.keras.optimizers import RMSprop

from .utils.categorical_distribution import CategoricalDistribution


class FCModel(BaseModel):
    """
    We expect users to code their own model.
    This one expects a box as observation and a discrete space for actions
    """

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            model_config,
    ):
        super(FCModel, self).__init__(
            name="FCModel",
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
        )
        self.output_size = action_space.n
        self.optimiser = RMSprop(
            learning_rate=model_config.lr,
            rho=model_config.rms_prop_rho,
            epsilon=model_config.rms_prop_epsilon
        )
        self.action_dist = CategoricalDistribution

        self._mlp = snt.nets.MLP(
            output_sizes=self.config.fc_dims,
            activate_final=True,
            name="MLP"
        )
        self._pi_out = snt.Linear(
            output_size=self.output_size,
            name="action_logits"
        )
        self._value_out = snt.Linear(
            output_size=self.output_size,
            name="values"
        )

    def forward(
            self,
            *,
            obs,
            **kwargs,
    ):
        x = self._mlp(obs)

        pi_out = self._pi_out(x)
        self._values = self._value_out(x)

        return pi_out, None

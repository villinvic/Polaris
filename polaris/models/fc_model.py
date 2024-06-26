import sonnet as snt
import numpy as np
from gymnasium.spaces import Discrete
import tensorflow as tf
from .base import BaseModel
from tensorflow.keras.optimizers import RMSprop

from .utils.categorical_distribution import CategoricalDistribution
from .. import SampleBatch


class FCModel(BaseModel):
    """
    We expect users to code their own model.
    This one expects a box as observation and a discrete space for actions
    """

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(FCModel, self).__init__(
            name="FCModel",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.output_size = action_space.n
        self.optimiser = RMSprop(
            learning_rate=config.lr,
            rho=config.rms_prop_rho,
            epsilon=config.rms_prop_epsilon
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
            output_size=1,
            name="values"
        )

    def initialise(self):

        T = 5
        B = 3
        x = self.observation_space.sample()
        dummy_obs = np.zeros_like(x, shape=(T, B) + x.shape)

        @tf.function
        def run(d):
            self(
                d
            )
        run({
                    SampleBatch.OBS: dummy_obs,
                 })

    def forward(
            self,
            *,
            obs,
            **kwargs,
    ):
        x = self._mlp(obs)

        pi_out = self._pi_out(x)
        self._values = tf.squeeze(self._value_out(x))

        return pi_out, None


class FCModelNoBias(BaseModel):
    """
    We expect users to code their own model.
    This one expects a box as observation and a discrete space for actions
    """

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(FCModelNoBias, self).__init__(
            name="FCModelNoBias",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.output_size = action_space.n
        self.optimiser = RMSprop(
            learning_rate=config.lr,
            rho=config.rms_prop_rho,
            epsilon=config.rms_prop_epsilon
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
            output_size=1,
            name="values",
            b_init=tf.zeros
        )

    def initialise(self):

        T = 5
        B = 3
        x = self.observation_space.sample()
        dummy_obs = np.zeros_like(x, shape=(T, B) + x.shape)

        @tf.function
        def run(d):
            self(
                d
            )
        run({
                    SampleBatch.OBS: dummy_obs,
                 })

    def forward(
            self,
            *,
            obs,
            **kwargs,
    ):
        x = self._mlp(obs)

        pi_out = self._pi_out(x)
        self._values = tf.squeeze(self._value_out(x))

        return pi_out, None


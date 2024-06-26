import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from .. import SampleBatch

tf.compat.v1.enable_eager_execution()


from .base import BaseModel
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.keras.layers import LSTM as tfLSTM

from .utils.categorical_distribution import CategoricalDistribution



class LSTM(snt.Module):
    def __init__(self, size, name):
        super().__init__(name)

        self._lstm = tfLSTM(

        )


class LSTMModel(BaseModel):
    """
    We expect users to code their own model.
    This one expects a box as observation and a discrete space for actions
    """

    is_recurrent = True


    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(LSTMModel, self).__init__(
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

        self._lstm = snt.LSTM(self.config.lstm_size, name="lstm")

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
        dummy_reward = np.zeros((T, B), dtype=np.float32)
        dummy_actions = np.zeros((T, B), dtype=np.int32)

        dummy_state = self.get_initial_state()
        states = [np.zeros_like(dummy_state, shape=(B,)+d.shape) for d in dummy_state]
        seq_lens = np.ones((B,), dtype=np.int32) * T

        @tf.function
        def run(d):
            self(
                d
            )
        run({
                    SampleBatch.OBS: dummy_obs,
                    SampleBatch.PREV_ACTION: dummy_actions,
                    SampleBatch.PREV_REWARD: dummy_reward,
                    SampleBatch.STATE: states,
                    SampleBatch.SEQ_LENS: seq_lens,
                 })

    def forward(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens,
            **kwargs,
    ):

        x = self._mlp(obs)
        lstm_input = tf.concat([
            x,
            tf.one_hot(prev_action, self.output_size, dtype=tf.float32),
            tf.tanh(tf.expand_dims(tf.cast(prev_reward, dtype=tf.float32), axis=-1)/5.),
        ], axis=-1)

        hidden, cell = tf.split(state, 2)

        # TODO: make everything time major ?
        lstm_out, states_out = snt.static_unroll(
            self._lstm,
            input_sequence=lstm_input,
            initial_state=snt.LSTMState(
                hidden=hidden[0],
                cell=cell[0]
            ),
            sequence_length=seq_lens,
        )

        states_out = [states_out.hidden, states_out.cell]

        pi_out = self._pi_out(lstm_out)
        self._values = tf.squeeze(self._value_out(lstm_out))

        return pi_out, states_out

    def get_initial_state(self):
        return [np.zeros((self.config.lstm_size,), dtype=np.float32) for _ in range(2)]


    def compute_action(
            self,
            input_dict: SampleBatch
    ):
        states = input_dict.pop(SampleBatch.STATE)
        batch_input_dict = tree.map_structure(lambda v: tf.expand_dims([v], axis=0), input_dict)
        batch_input_dict[SampleBatch.SEQ_LENS] = tf.expand_dims(1, axis=0)
        batch_input_dict[SampleBatch.STATE] = [tf.expand_dims(state, axis=0) for state in states
                                               ]

        action_logits, state = self._compute_action_dist(
            batch_input_dict
        )

        action_logits = tf.squeeze(action_logits).numpy()
        action_dist = self.action_dist(action_logits)
        action = action_dist.sample()
        logp = action_dist.logp(action).numpy()
        return action.numpy(), state, logp, action_logits


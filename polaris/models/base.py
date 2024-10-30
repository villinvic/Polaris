import sonnet as snt
import tree
from gymnasium.spaces import Space, Dict, Box, Discrete
from ml_collections import ConfigDict
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import time

from polaris.experience import SampleBatch



def expand_values(v):
    if v is None:
        return v
    else:
        return tf.expand_dims([v], axis=0)

class BaseModel(snt.Module):
    is_recurrent = False

    def __init__(
            self,
            name: str,
            observation_space: Space,
            action_space: Space,
            config: ConfigDict,
    ):
        super(BaseModel, self).__init__(name=name)

        self.action_space = action_space
        self.observation_space = observation_space
        self.num_outputs = None
        self.config = config
        self.optimiser: Optimizer = None

        self._values = None
        self.action_dist = None


    @snt.once
    def setup(self):
        self.initialise()

    def __call__(self, input_dict):

        return self.forward(**input_dict)


    def initialise(self):
        """
        Initialise the layers and model related stuffs
        """
        pass

    def forward(self, **input_dict):
        """
        Does a pass forward
        :param input_dict: data needed to make a pass forward
        :return: main output of the model -> actions
        """
        raise NotImplementedError


    def value_function(self):
        """

        :return: the values predicted from the previous pass
        """
        return self._values

    def get_initial_state(self):

        return None


    def prepare_single_input(self, input_dict: SampleBatch):
        input_dict[SampleBatch.SEQ_LENS] = tf.expand_dims(1, axis=0)
        input_dict["single_obs"] = True
        return input_dict


    def compute_action(
            self,
            input_dict: SampleBatch
    ):

        t = time.time()
        #batch_input_dict = tree.map_structure(expand_values, input_dict)
        self.prepare_single_input(input_dict)


        (action_logits, state), value, action, logp, extras = self._compute_action_dist(
            input_dict
        )

        out = (action.numpy(), tree.map_structure(lambda v: v.numpy(), state), logp.numpy(),
               action_logits.numpy(), value.numpy())


        extras["compute_action_ms"] = time.time() - t

        return out + (extras,)

    def compute_value(self, input_dict: SampleBatch):
        self.prepare_single_input(input_dict)
        return self._compute_value(input_dict).numpy()


    @tf.function
    def _compute_value(self, input_dict):
        _, value = self(input_dict)
        return value


    @tf.function
    def _compute_action_dist(self, input_dict):
        (action_logits, state), value, extras = self(input_dict)
        action_logits = tf.squeeze(action_logits)
        action_dist = self.action_dist(action_logits)
        action = action_dist.sample()
        logp = action_dist.logp(action)
        return (action_logits, state), value, action, logp, extras




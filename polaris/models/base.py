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


    def compute_action(
            self,
            input_dict: SampleBatch
    ):

        tx = []
        t = time.time()
        #batch_input_dict = tree.map_structure(expand_values, input_dict)
        input_dict[SampleBatch.SEQ_LENS] = tf.expand_dims(1, axis=0)
        input_dict["single_obs"] = True

        tx.append(time.time() - t)
        t = time.time()


        (action_logits, state), value, action, logp = self._compute_action_dist(
            input_dict
        )

        tx.append(time.time() - t)
        t = time.time()

        out = (action.numpy(), tree.map_structure(lambda v: v.numpy(), state), logp.numpy(),
               action_logits.numpy(), value.numpy())

        tx.append(time.time() - t)

        return out + (tx,)


    @tf.function
    def _compute_action_dist(self, input_dict):
        (action_logits, state), value = self(input_dict)
        action_logits = tf.squeeze(action_logits)
        action_dist = self.action_dist(action_logits)
        action = action_dist.sample()
        logp = action_dist.logp(action)
        return (action_logits, state), value, action, logp


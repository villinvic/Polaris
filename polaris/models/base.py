import sonnet as snt
from gymnasium.spaces import Space, Dict, Box, Discrete
from ml_collections import ConfigDict
from tensorflow.keras.optimizers import Optimizer


class BaseModel(snt.Module):

    def __init__(
            self,
            name: str,
            observation_space: Space,
            action_space: Space,
            model_config: ConfigDict,
    ):
        super(BaseModel, self).__init__(name=name)

        self.action_space = action_space
        self.observation_space = observation_space
        self.num_outputs = None
        self.config = model_config
        self.optimiser: Optimizer = None

        self._values = None
        self.action_dist = None

    @snt.once
    def _setup(self, input_dict):
        self._initialise(input_dict)

    def __call__(self, input_dict):
        self._setup(input_dict)
        return self.forward(**input_dict)


    def _initialise(self, input_dict):
        """
        Initialise the layers and model related stuffs

        :param input_dict:
        :return:
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


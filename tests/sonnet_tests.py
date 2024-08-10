import sonnet as snt
import tensorflow as tf
import numpy as np


class BaseModel(snt.Module):
    is_recurrent = False

    def __init__(
            self,
            name = "test"

    ):
        super(BaseModel, self).__init__(name=name)

        self.c = tf.Variable(0)




x = BaseModel()

print(x.variables)


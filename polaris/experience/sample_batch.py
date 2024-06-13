from copy import deepcopy

import numpy as np
from ml_collections import ConfigDict
import tree

class SampleBatch(dict):

    OBS = "obs"
    PREV_OBS = "prev_obs"
    ACTION = "action"
    PREV_ACTION = "prev_action"
    REWARD = "reward"
    DONE = "done"
    EPISODE_ID = "episode_id"
    POLICY_ID = "policy_id"
    AGENT_ID = "agent_id"
    STATE = "state"

    def __init__(
            self,
            batch_size,
            # float_dtype=np.float32,
            # int_dtype=np.int32
    ):
        super().__init__()

        self.batch_size = batch_size
        self.index = 1
        # self.float_dtype = float_dtype
        # self.int_dtype = int_dtype

    # def __getitem__(self, key):
    #     if "prev" in key:
    #         key = key.lstrip("prev_")
    #         return super().__getitem__(key)[:-1]
    #     else:
    #         print(key, self)
    #         return super().__getitem__(key)[1:]


    def init_key(self, key, value):
        def leaves_to_numpy(value):
            if isinstance(value, np.ndarray):
                return np.zeros((self.batch_size + 1,) + value.shape, dtype=value.dtype)
            elif isinstance(value, np.float32):
                return np.zeros((self.batch_size + 1,), dtype=np.float32)
            elif isinstance(value, np.int32):
                return np.zeros((self.batch_size + 1,), dtype=np.int32)
            elif isinstance(value, str):
                return np.zeros((self.batch_size + 1,), dtype=str)
            else:
                return np.zeros((self.batch_size + 1,), dtype=object)
        self[key] = tree.map_structure(leaves_to_numpy, value)

    def advance(self):
        self.index += 1

    def reset(self):
        self.index = 1

    def push(self, data):
        for key, value in data.items():
            if key not in self:
                self.init_key(key, value)
            print(key, self[key])
            self[key][self.index] = value
        self.advance()
        if self.is_full():
            self.reset()
            return [self]
        return []
        # when full, reset and send batch

    def is_full(self):
        return self.batch_size == self.index

    def __getslice__(self, key: str, i, j):
        if not key.startswith("prev_"):
            i += 1
            j += 1
        else:
            key = key.lstrip("prev_")

        return tree.map_structure(lambda x: x[i: j], self[key])

    def __repr__(self):
        return f"SampleBatch(size={self.batch_size}, content={list(self.keys())})"


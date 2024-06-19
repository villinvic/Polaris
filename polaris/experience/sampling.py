from typing import List

import numpy as np
from ml_collections import ConfigDict
import tree

class SampleBatch(dict):

    OBS = "obs"
    PREV_OBS = "prev_obs"
    NEXT_OBS = "next_obs"
    ACTION = "action"
    ACTION_LOGP = "action_logp"
    ACTION_LOGITS = "action_logits"
    PREV_ACTION = "prev_action"
    REWARD = "reward"
    PREV_REWARD = "prev_reward"
    DONE = "done"
    EPISODE_ID = "episode_id"
    POLICY_ID = "policy_id"
    AGENT_ID = "agent_id"
    STATE = "state"
    NEXT_STATE = "next_state"
    T = "t"
    VERSION = "version"

    def __init__(
            self,
            batch_size,
            # float_dtype=np.float32,
            # int_dtype=np.int32
    ):
        super().__init__()

        self.batch_size = batch_size
        self.index = 0
        self.batch_dim = 1
        self.episode_slicing_indices = None
        self.seq_lens = None

        # Is this nice ?
        self.compression = False
        # self.float_dtype = float_dtype
        # self.int_dtype = int_dtype

    def __getitem__(self, key):
        if self.compression:
            if key not in self :
                if key.startswith("prev_"):
                    key = key.lstrip("prev_")
                    return tree.map_structure(
                        lambda v: v[:-2],
                        super().__getitem__(key)
                    )
                elif key.startswith("next_"):
                    key = key.lstrip("prev_")
                    return tree.map_structure(
                        lambda v: v[2:],
                        super().__getitem__(key)
                    )
            else:

                return tree.map_structure(
                    lambda v: v[1:-1],
                    super().__getitem__(key)
                )
        else:
            return super().__getitem__(key)

    def init_key(self, key, value):

        # + 2
        def leaves_to_numpy(value):
            if isinstance(value, np.ndarray):
                return np.zeros((1, self.batch_size,) + value.shape, dtype=value.dtype)
            elif isinstance(value, np.float32):
                return np.zeros((1, self.batch_size,), dtype=np.float32)
            elif isinstance(value, np.int32):
                return np.zeros((1, self.batch_size,), dtype=np.int32)
            elif isinstance(value, bool):
                return np.zeros((1, self.batch_size,), dtype=np.bool_)
            elif isinstance(value, str):
                return np.zeros((1, self.batch_size,), dtype='<U20')
            else:
                return np.zeros((1, self.batch_size,), dtype=object)
        self[key] = tree.map_structure(leaves_to_numpy, value)

    def advance(self):
        self.index += 1

    def reset(self):
        # tree.map_structure(
        #     self.set_previous,
        #     self
        # )
        self.index = 0

    def push(self, data):
        if self.is_full():
            self.reset()

        for key, value in data.items():
            if key not in self:
                self.init_key(key, value)

            tree.map_structure(
                self.set_item,
                super().__getitem__(key), value
            )
        self.advance()
        if self.is_full():
            return [self]
        return []
        # when full, reset and send batch

    def is_full(self):
        return self.batch_size == self.index

    def __getslice__(self, key: str, i, j):
        if self.compression:
            if not key.startswith("prev_"):
                i += 1
                j += 1
            else:
                key = key.lstrip("prev_")

        return tree.map_structure(lambda x: x[i: j], self[key])


    def __repr__(self):
        return f"SampleBatch(size={self.batch_size}, content={list(self.keys())})"

    def set_item(self, batch, v):
        batch[:, self.index] = v

    def set_previous(self, batch):

        # -1 is for "next obs"
        batch[:, 0] = batch[:, -1]

    def get_owner(self):
        return super().__getitem__("policy_id")[0]

    def compute_episode_slicing_indices(self):
        # TODO: either run this in a loop, or at epsiode generation time
        if self.episode_slicing_indices is None:
            indices = []
            for batch_id in range(self.batch_dim):
                diff = np.diff(self[SampleBatch.EPISODE_ID][batch_id])
                batch_indices = np.argwhere(diff!=0)[:, 0] +1

                indices.append(batch_indices)

            self.episode_slicing_indices = indices

    def split_by_episode(self, key):
        self.compute_episode_slicing_indices()
        return np.split(self[key], self.episode_slicing_indices)

    def get_seq_lens(self, max_seq_lens):
        self.compute_episode_slicing_indices()
        if self.seq_lens is None:
            seq_lens = []
            for batch_id in range(self.batch_dim):
                new_episode_indices = [0] + list(self.episode_slicing_indices[batch_id]) + [self.batch_size]
                batch_seq_lens = []
                for i, index in enumerate(new_episode_indices[:-1]):
                    delta = new_episode_indices[i+1] - index
                    count = 0

                    while count < delta:
                        seq_len = min(max_seq_lens, delta-count)
                        batch_seq_lens.append(seq_len)
                        count += seq_len
                seq_lens.append(np.array(batch_seq_lens, dtype=np.int32))
            self.seq_lens = seq_lens

        return self.seq_lens




def concat_sample_batches(batches: List[SampleBatch]):
    """
    Concatenates the batches together, note that the batche
    :param batches: Batches to concatenate
    :return:
    """
    batch_dim = sum(b.batch_dim for b in batches)

    new_batch = SampleBatch(batches[0].batch_size)
    new_batch.compression = False
    new_batch.batch_dim = batch_dim


    all_keys = {*batches[0].keys()}

    # TODO : how should we handle prev obs, etc. ?
    #        the problem comes from the fact that we cant concat two batches as of right now.
    #        instantiate two different columns ? we could...
    for key in all_keys:
        new_batch[key] = tree.map_structure(
            lambda *b: np.concatenate(b, axis=0), # np.insert(np.concatenate(b, axis=0), 0, 0, axis=0)
            *(bb[key] for bb in batches)
        )


    return new_batch


class ExperienceQueue:

    def __init__(
            self,
            config: ConfigDict,
    ):
        self.config = config
        self.queue = []

    def push(self, batches: List[SampleBatch]):
        if len({b.get_owner() for b in batches}) != 1:
            raise ValueError("Pushing to queue experience from different policies!")

        self.queue.extend(batches)
        if len(self.queue) > self.config.max_queue_size:
            print(f"Experience queue is too long !: {len(self.queue)} samples waiting.")

    def pull(self, num_samples):

        if num_samples % self.config.batch_size != 0:
            raise ValueError(f"The number of samples pulled from the queue must be a multiple of the batch_size. "
                             f"Batch size = {self.config.batch_size}, Number of samples requested = {num_samples}")
        assert self.is_ready()
        num_batches = num_samples // self.config.batch_size
        samples = self.queue[:num_samples]
        self.queue = self.queue[num_batches:]
        return concat_sample_batches(samples)

    def is_ready(self):
        return len(self.queue) * self.config.batch_size >= self.config.train_batch_size


if __name__ == '__main__':

    q = SampleBatch(20)
    q2 = SampleBatch(20)
    for i in range(20):
        q.push({SampleBatch.OBS   : np.array([1, 2]), SampleBatch.ACTION: 2, SampleBatch.EPISODE_ID: i // 4,
                SampleBatch.REWARD: np.float32(np.random.randint(5)), SampleBatch.DONE: np.random.random() < 0.1})
    for i in range(20):
        q2.push({SampleBatch.OBS   : np.array([1, 2]), SampleBatch.ACTION: 2, SampleBatch.EPISODE_ID: i // 5,
                SampleBatch.REWARD: np.float32(np.random.randint(5)), SampleBatch.DONE: np.random.random() < 0.1})

    q = concat_sample_batches([q, q2])
    q.compute_episode_slicing_indices()
    print(q.episode_slicing_indices)
    print(q.get_seq_lens(3))
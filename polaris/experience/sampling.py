import copy
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

    VALUES = "values"
    ADVANTAGES = "advantages"
    VF_TARGETS = "vf_targets"


    SEQ_LENS = "seq_lens" # Does not have the time dimension

    def __init__(
            self,
            trajectory_length,
            max_seq_len=None,
            **initial_batch
    ):
        super().__init__(**initial_batch)

        self.trajectory_length = trajectory_length
        self.index = 0
        self.sequence_counter = 0
        self.max_seq_len = max_seq_len if max_seq_len is not None else trajectory_length
        self[SampleBatch.SEQ_LENS] = []
        self[SampleBatch.STATE] = []


    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__getslice__(key)
        return super().__getitem__(key)

    def init_key(self, key, value):

        # + 2
        def leaves_to_numpy(value):
            if isinstance(value, np.ndarray):
                return np.zeros((self.trajectory_length,) + value.shape, dtype=value.dtype)
            elif isinstance(value, (np.float32, float)) or value is None:
                return np.zeros((self.trajectory_length,), dtype=np.float32)
            elif isinstance(value, (np.int32, np.int16, int)):
                return np.zeros((self.trajectory_length,), dtype=np.int32)
            elif isinstance(value, bool):
                return np.zeros((self.trajectory_length,), dtype=np.bool_)
            elif isinstance(value, str):
                return np.zeros((self.trajectory_length,), dtype='<U20')
            else:
                return np.zeros((self.trajectory_length,), dtype=object)
        self[key] = tree.map_structure(leaves_to_numpy, value)

    def advance(self):
        self.index += 1

    def reset(self):
        self.index = 0
        self.sequence_counter = 0
        self[SampleBatch.SEQ_LENS] = []
        self[SampleBatch.STATE] = []

    def size(self):
        return self.trajectory_length

    def push(self, data, flush=False):
        if self.is_full():
            self.reset()

        for key, value in data.items():
            if key not in self:
                self.init_key(key, value)

            if key == self.STATE:
                if self.index % self.max_seq_len == 0:
                    self[SampleBatch.STATE].append(copy.deepcopy(value))
            elif key == self.NEXT_STATE:
                if self.index == (self.trajectory_length-1):
                    self[SampleBatch.NEXT_STATE] = copy.deepcopy(value)
            else:
                tree.map_structure(
                    self.set_item,
                    super().__getitem__(key), value
                )

        done = data.get(SampleBatch.DONE, False)
        self.advance()

        self.sequence_counter += 1
        if done or self.sequence_counter % self.max_seq_len == 0 or self.is_full():
            # TODO check here ???
            if self.sequence_counter % self.max_seq_len == 0:
                seq_len = self.max_seq_len
            else:
                seq_len = self.sequence_counter % self.max_seq_len
                self.sequence_counter = 0
            self[SampleBatch.SEQ_LENS].append(seq_len)

        if self.is_full():
            assert sum(self[SampleBatch.SEQ_LENS])==self.trajectory_length, f"Seq_lens not summing to {self.trajectory_length}"
            return [self]
        elif flush:
            assert done, "flushing but not done ?"
            # TODO: on hold
            # TODO: what happens when seq_lens != traj len
            # we fill the remaining with dones and appropriate seq len
            #remaining = self.index - sum(self[SampleBatch.SEQ_LENS])
            self[SampleBatch.POLICY_ID][self.index:] = self[SampleBatch.POLICY_ID][0]
            for k, v in self.items():
                if k == SampleBatch.POLICY_ID:
                    v[self.index:] = self[SampleBatch.POLICY_ID][0]
                elif k == SampleBatch.DONE:
                    v[self.index:] = True
                elif k == SampleBatch.REWARD:
                    v[self.index:] = 0.
                elif k == SampleBatch.PREV_REWARD:
                    v[self.index:] = 0.
                elif k == SampleBatch.VERSION:
                    v[self.index:] = self[SampleBatch.VERSION][0]

            self.index = self.trajectory_length

            # last_seq_len = self.index - sum(self[SampleBatch.SEQ_LENS])
            # self[SampleBatch.SEQ_LENS].append(last_seq_len)
            # truncated = self[:self.index]
            # self.index = self.trajectory_length
            return [self]
        return []
        # when full, reset and send batch

    def is_full(self):
        return self.trajectory_length == self.index

    def __getslice__(self, s: slice):

        i = s.start*self.max_seq_len if s.start is not None else None
        j = s.stop*self.max_seq_len if s.stop is not None else None
        step = s.step if s.step is not None else None
        s_seq = slice(i,j,step)

        d = dict(self)
        seq_lens = d.pop(SampleBatch.SEQ_LENS)
        state = d.pop(SampleBatch.STATE)
        next_state = d.pop(SampleBatch.NEXT_STATE, None)


        sliced_batch = SampleBatch(trajectory_length=self.trajectory_length, max_seq_len=self.max_seq_len, **tree.map_structure(lambda x: x[s_seq], d))
        sliced_batch.trajectory_length = len(sliced_batch[SampleBatch.REWARD])
        sliced_batch[SampleBatch.SEQ_LENS] = seq_lens[s]
        sliced_batch[SampleBatch.STATE] = tree.map_structure(lambda x: x[s], state)
        if next_state is not None:
            sliced_batch[SampleBatch.NEXT_STATE] = tree.map_structure(lambda x: x[s], next_state)

        return sliced_batch


    def __repr__(self):
        return f"SampleBatch(size={self.trajectory_length}, content={list(self.keys())})"

    def set_item(self, batch, v):
        batch[self.index] = v

    def get_owner(self):
        #assert np.all(super().__getitem__("policy_id")[0] == super().__getitem__("policy_id")), super().__getitem__("policy_id")
        return super().__getitem__(SampleBatch.POLICY_ID)[0]

    def get_aid(self):
        return super().__getitem__(SampleBatch.AGENT_ID)[0]

    def pad_sequences(self):
        """
        :return: the batch split and padded into sequence bits
        """
        seq_lens = self[SampleBatch.SEQ_LENS]
        missing = self.trajectory_length - np.sum(seq_lens)
        if missing == 0:
            return self

        missing_bits = (self.trajectory_length - len(seq_lens) * self.max_seq_len)//self.max_seq_len
        # we could just flush the queue instead, here we may just miss some relevant samples
        seq_lens += [0] * missing_bits

        states = self[SampleBatch.STATE]
        states += [states[-1]] * missing_bits

        # split_indices = np.cumsum(seq_lens)[:-1]
        #
        # def get_padding(element, seq_len):
        #     rank = len(element.shape)
        #     p = ((0, self.max_seq_len-seq_len),) + ((0,0),) * (rank - 1)
        #     #print(np.pad(element, p).shape, element.shape, seq_len)
        #     return ((0, self.max_seq_len-seq_len),) + ((0,0),) * (rank - 1)
        #
        # def split_pad_concat(full_seq):
        #     sequences = np.split(full_seq, split_indices, axis=0)
        #     print(split_indices)
        #     print([s.shape for s in sequences])
        #     return np.concatenate([
        #         np.pad(seq, get_padding(seq, seq_len))
        #         for seq, seq_len in zip(sequences, seq_lens)
        #     ])
        #
        # d = dict(self)
        # seq_lens = d.pop(SampleBatch.SEQ_LENS)
        # d.pop("bootstrap_value")
        # d.pop(SampleBatch.NEXT_STATE)
        # new_batch = SampleBatch(trajectory_length=self.trajectory_length, max_seq_len=self.max_seq_len, **tree.map_structure(
        #     split_pad_concat,
        #     d
        # ))
        # new_batch[SampleBatch.SEQ_LENS] = seq_lens
        #
        # print("action?",new_batch[SampleBatch.ACTION].shape)
        #
        # print(seq_lens, self.max_seq_len*len(seq_lens), new_batch.trajectory_length)

        self[SampleBatch.SEQ_LENS] = seq_lens

        return self


def concat_sample_batches(batches: List[SampleBatch]):
    """
    Concatenates the batches together, note that the batche
    :param batches: Batches to concatenate
    :return:
    """
    traj_len = sum(b.trajectory_length for b in batches)


    new_batch = SampleBatch(traj_len, max_seq_len=batches[0].max_seq_len)
    for key in batches[0]:
        def p(*b):
            return np.concatenate(b, axis=0)

        if key == SampleBatch.SEQ_LENS:
            new_batch[SampleBatch.SEQ_LENS] = np.concatenate([
                bb[SampleBatch.SEQ_LENS] for bb in batches
            ], axis=0)
        elif key == SampleBatch.STATE:
            # flatten out the lists
            states = []
            for batch in batches:
                state = batch[key]
                if isinstance(state, list):
                    states.extend(state)
                else:
                    states.append(state)

            new_batch[key] = tree.map_structure(
                p,
                *states
            )
        elif key in ("bootstrap_value", SampleBatch.NEXT_STATE):
            pass
        else:
            new_batch[key] = tree.map_structure(
                p,
                *(bb[key] for bb in batches)
            )

    return new_batch



class ExperienceQueue:

    def __init__(
            self,
            config: ConfigDict,
    ):
        self.config = config
        self.queue: SampleBatch = None

        self.seq_len = self.config.max_seq_len if self.config.max_seq_len is not None else self.config.trajectory_length

        self.last_batch = None

    def push(self, batches: List[SampleBatch]):
        if len({b.get_owner() for b in batches}) != 1:
            raise ValueError("Pushing to queue experience from different policies!")

        if self.queue is None:
            self.queue = concat_sample_batches(batches)
        else:
            self.queue = concat_sample_batches([self.queue]+batches)

        if self.queue.size() > self.config.max_queue_size:
            print(f"Experience queue is too long !: {self.queue.size()} samples waiting.")

    def pull(self, num_samples):
        # if num_batches % self.config.batch_size != 0:
        #     raise ValueError(f"The number of samples pulled from the queue must be a multiple of the batch_size. "
        #                      f"Batch size = {self.config.batch_size}, Number of samples requested = {num_samples}")
        assert self.is_ready()

        assert num_samples % self.seq_len == 0, (num_samples, self.seq_len)
        num_batches = num_samples // self.seq_len

        samples = self.queue[:num_batches]
        self.queue = self.queue[num_batches:]

        self.last_batch = samples
        #print(f"pulling {samples.size()} samples, {self.queue.size()} samples left, real size {len(samples[SampleBatch.ACTION])}")

        return samples


    def is_ready(self):
        return self.queue is not None and (self.queue.size() >= self.config.train_batch_size)

    def size(self):
        return 0 if self.queue is None else self.queue.size()
    def get_epochs(self, n_epochs):
        for k in range(n_epochs):

            ordering = np.arange(self.config.train_batch_size // self.config.minibatch_size)
            np.random.shuffle(ordering)
            minibatch_indices = np.split(ordering, self.config.minibatch_size//self.queue.max_seq_len)
            for indices in minibatch_indices:

                yield self.queue.get_from_indices(indices)




def get_epochs(time_major_batch, n_epochs, minibatch_size):
    max_seq_len, n_trajectories = time_major_batch[SampleBatch.ACTION].shape[:2]
    seq_lens = np.array(time_major_batch.pop(SampleBatch.SEQ_LENS))
    state = time_major_batch.pop(SampleBatch.STATE)
    next_state = time_major_batch.pop(SampleBatch.NEXT_STATE, None)

    batch_size = max_seq_len * n_trajectories

    ordering = np.arange(n_trajectories)
    for k in range(n_epochs):
        np.random.shuffle(ordering)
        minibatch_indices = np.split(ordering, batch_size//minibatch_size)
        for indices in minibatch_indices:
            def f(d):
                return d[:, indices]

            minibatch =  tree.map_structure(
                f,
                time_major_batch
            )
            minibatch[SampleBatch.SEQ_LENS] = seq_lens[indices]
            minibatch[SampleBatch.STATE] = tree.map_structure(lambda x: x[indices], state)
            if next_state is not None:
                minibatch[SampleBatch.NEXT_STATE] = tree.map_structure(lambda x: x[indices], next_state)

            #print(minibatch[SampleBatch.ACTION].shape)
            yield minibatch

if __name__ == '__main__':

    q = SampleBatch(20, max_seq_len=5)
    q2 = SampleBatch(20, max_seq_len=5)
    ep_id = 0
    t = 0
    for i in range(20):
        done = np.random.random() < 0.1
        q.push({SampleBatch.OBS   : np.array([1, 2]), SampleBatch.ACTION: 2, SampleBatch.EPISODE_ID: ep_id,
                SampleBatch.REWARD: np.float32(np.random.randint(5)), SampleBatch.DONE: done,
                SampleBatch.T : t})
        t += 1
        if done:
            ep_id += 1
            t = 0

    t = 0
    ep_id = 25
    for i in range(20):
        done = np.random.random() < 0.1

        q2.push({SampleBatch.OBS   : np.array([1, 2]), SampleBatch.ACTION: 2, SampleBatch.EPISODE_ID: ep_id,
                SampleBatch.REWARD: np.float32(np.random.randint(5)), SampleBatch.DONE: done, SampleBatch.T : t})
        t += 1
        if done:
            t = 0
            ep_id += 1
    q = concat_sample_batches([q, q2])
    padded_batch = q.pad_and_split_to_sequences()
    print(padded_batch[SampleBatch.DONE], padded_batch[SampleBatch.SEQ_LENS])
    trunc_batch = padded_batch[:5]
    print(trunc_batch[SampleBatch.DONE], trunc_batch[SampleBatch.SEQ_LENS])
    exit()
    q = concat_sample_batches([q, q2])
    q.compute_episode_slicing_indices()
    print(q.episode_slicing_indices)
    print(q.get_seq_lens(3))
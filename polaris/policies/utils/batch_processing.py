import numpy as np
from polaris.experience import SampleBatch
import tree


def make_time_major(b):
    """
    Reshapes the batch so that the trajectories get arranged such that the batch components are of shape [T, B, ...]
    The SEQ_LENS component of the batch provides the lengths of the B sequences.
    The STATE component provides the state of the policy at the beginning of the B sequences.
    """

    b = dict(b)

    seq_lens = b.pop(SampleBatch.SEQ_LENS)
    num_sequences = len(seq_lens)

    def make_time_major(p, v):
        try:
            return np.transpose(np.reshape(v, (num_sequences, -1) + v.shape[1:]),
                                (1, 0) + tuple(range(2, 1 + len(v.shape))))
        except Exception as e:
            print(p, v)
            raise e

    time_major_batch = tree.map_structure_with_path(make_time_major, b)

    time_major_batch[SampleBatch.SEQ_LENS] = seq_lens
    time_major_batch[SampleBatch.STATE] = tree.map_structure(lambda v: v[0],time_major_batch[SampleBatch.STATE])

    return time_major_batch
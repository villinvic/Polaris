import sonnet as snt
import tensorflow as tf
import numpy as np
core = snt.LSTM(hidden_size=16)
batch_size = 1
input_sequence = tf.random.uniform([32, batch_size, 2])
output_sequence, final_state = snt.static_unroll(
core,
input_sequence,
core.initial_state(batch_size),
sequence_length=np.array([[8, 16, 8]])
)

print(output_sequence, final_state)

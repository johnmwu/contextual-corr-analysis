import numpy as np 
from get_transformer_representations import make_hdf5_file


def make_attn_dataset(num_layers, num_heads, sequence_lengths, output_filename):

	idx_to_atten, sentence_to_idx = dict(), dict()
	for i, l in enumerate(sequence_lengths):
		idx_to_atten[str(i)] = np.random.randn(num_layers, num_heads, l, l)
		sentence_to_idx[str(i)] = str(i)

	make_hdf5_file(sentence_to_idx, idx_to_atten, output_filename)


sequence_lengths = [3, 3]
make_attn_dataset(2, 2, sequence_lengths, 'tests/attn1.hdf5')
make_attn_dataset(2, 2, sequence_lengths, 'tests/attn2.hdf5')
make_attn_dataset(3, 2, sequence_lengths, 'tests/attn3.hdf5')
make_attn_dataset(3, 3, sequence_lengths, 'tests/attn4.hdf5')



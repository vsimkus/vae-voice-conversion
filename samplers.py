import torch
import math
from torch.utils.data import Sampler

class ChunkEfficientRandomSampler(Sampler):
    r"""Chunk-efficient random sampler for large datasets that loads chunk-by-chunk.
    ChunkEfficientRandomSampler allows random sampling by randomizing the chunks 
    and within-chunk samples, without the overhead of completely random sampling,
    which requires the dataset to load a different chunk-file for each sample.

    This was created for efficiently randomizing VCTK dataset.

    Arguments:
        data_source (Dataset): dataset to sample indices for.
        chunk_len (int): length of a chunk-file of the dataset.
    """

    def __init__(self, data_source, chunk_len):
        self.data_source = data_source
        self.chunk_len = chunk_len
        self.num_chunks = math.ceil(len(data_source)/self.chunk_len)

        if not isinstance(self.chunk_len, int) or self.chunk_len <= 0:
            raise ValueError("chunk_len should be a positive integer " \
                             "value, but got chunk_len={}".format(self.chunk_len))

    def __iter__(self):
        # Prepare an index list for each chunk and randomize
        chunk_indices = torch.arange(0, self.num_chunks)
        chunk_indices = chunk_indices[torch.randperm(self.num_chunks)]

        for ch_i in chunk_indices:
            # Last chunk might be shorter so check for the length
            curr_chunk_len = self.chunk_len if ch_i != self.num_chunks-1 else len(self.data_source) % self.chunk_len
            # Prepare indices for the current chunk and randomize
            indices = torch.arange(ch_i*self.chunk_len, ch_i*self.chunk_len + curr_chunk_len)
            indices = indices[torch.randperm(curr_chunk_len)]
            for i in indices:
                yield(i)

    def __len__(self):
        return len(self.data_source)
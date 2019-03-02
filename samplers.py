import math
import numpy as np
from torch.utils.data import Sampler
from itertools import islice

class ChunkEfficientRandomSampler(Sampler):
    r"""Chunk-efficient random sampler for large datasets that loads chunk-by-chunk.
    ChunkEfficientRandomSampler allows random sampling by randomizing the chunks 
    and within-chunk samples, without the overhead of completely random sampling,
    which requires the dataset to load a different chunk-file for each sample.

    This was created for efficiently randomizing VCTK dataset.

    Arguments:
        data_source (Dataset): dataset to sample indices for.
        chunk_len (int): length of a chunk-file of the dataset.
        first_chunk_offset (int, default=0): integer offset of the first chunk. 
            For example when splitting a dataset into train and validation, 
            the split might occur in the middle of a chunk, so the offset 
            here lets us know what part of the chunk we should be starting at.
    """

    def __init__(self, data_source, chunk_len, first_chunk_offset=0):
        self.data_source = data_source
        self.chunk_len = chunk_len
        self.first_chunk_offset = first_chunk_offset
        self.num_chunks = math.ceil((len(self.data_source)+self.first_chunk_offset)/self.chunk_len)

        if not isinstance(self.chunk_len, int) or self.chunk_len <= 0:
            raise ValueError("chunk_len should be a positive integer " \
                             "value, but got chunk_len={}".format(self.chunk_len))
        
        if not isinstance(self.first_chunk_offset, int) or self.first_chunk_offset < 0:
            raise ValueError("first_chunk_offset should be a non-negative integer " \
                             "value, but got first_chunk_offset={}".format(self.first_chunk_offset))


    def __iter__(self):
        # Prepare a list of indices
        indices = np.arange(0, len(self.data_source))

        # Prepare a list of chunk-sizes TODO: refactor
        chunk_sizes = []
        total_indices = len(self.data_source)
        for ch_i in range(self.num_chunks):
            if ch_i == 0: # First chunk might be smaller due to offset
                chunk_size = self.chunk_len - self.first_chunk_offset
            elif ch_i == self.num_chunks-1: # Final chunk should have all the rest of indices
                chunk_size = total_indices
            else: # Intermediate chunks are full-length
                chunk_size = self.chunk_len
            
            chunk_sizes.append(chunk_size)
            total_indices -= chunk_size
        
        # Slice index list according to chunk sizes
        index_it = iter(indices)
        chunked_indices = [np.array(list(islice(index_it, 0, i))) for i in chunk_sizes]
        chunked_indices = np.array(chunked_indices)

        # Permute chunks
        chunked_indices = np.random.permutation(chunked_indices)
        for chunk in chunked_indices:
            # Permute indices in chunk
            for i in np.random.permutation(chunk):
                yield i

    def __len__(self):
        return len(self.data_source)
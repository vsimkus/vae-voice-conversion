import math
import numpy as np
from torch.utils.data import Sampler
from itertools import islice
import random

class ChunkEfficientRandomSampler(Sampler):
    r"""Chunk-efficient random sampler for large datasets that loads chunk-by-chunk.
    ChunkEfficientRandomSampler allows random sampling by randomizing the chunks 
    and within-chunk samples, without the overhead of completely random sampling,
    which requires the dataset to load a different chunk-file for each sample.

    This was created for efficiently randomizing VCC2016 dataset.

    Arguments:
        data_source (Dataset): dataset to sample indices for.
        chunk_indices (map:(tuple)): start and end indices for each chunk.
    """

    def __init__(self, data_source, chunk_indices):
        self.data_source = data_source
        self.chunk_indices = chunk_indices

    def __iter__(self):
        # Prepare a list of indices
        chunk_keys = [*self.chunk_indices.keys()]
        random.shuffle(chunk_keys)
        
        for key in chunk_keys:
            chunk = self.chunk_indices[key]
            chunk_min = chunk[0]
            chunk_max = chunk[1]
            indices = np.arange(chunk_min, chunk_max+1)
            for i in np.random.permutation(indices):
                yield i

    def __len__(self):
        return len(self.data_source)
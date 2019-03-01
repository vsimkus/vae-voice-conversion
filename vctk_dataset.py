import torch.utils.data as data
import numpy as np
import os
import os.path
import torch

class VCTKDataset(data.Dataset):
    """`VCTK <http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`_ Dataset.
    `alternate url <http://datashare.is.ed.ac.uk/handle/10283/2651>`
    Based on torchaudio vctk.py <https://github.com/pytorch/audio>

    Args:
        root (string): Root directory of dataset where ``vctk/processed/training.pt``
            and  ``vctk/processed/test.pt`` exist.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Scale``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    raw_folder = 'vctk/raw'
    processed_folder = 'vctk/processed'
    zip_path = 'VCTK-Corpus.zip'  # path to local zip file
    dset_path = 'VCTK-Corpus'

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.labels = []
        self.chunk_size = 1000
        self.num_samples = 0
        self.max_len = 0
        self.mean_len = 0.
        self.std_len = 0.
        self.cached_pt = 0

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        self._read_info()
        self.data, self.labels = torch.load(os.path.join(
            self.root, self.processed_folder, "vctk_{:04d}.pt".format(self.cached_pt)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            self.data, self.labels = torch.load(os.path.join(
                self.root, self.processed_folder, "vctk_{:04d}.pt".format(self.cached_pt)))
        index = index % self.chunk_size
        audio, target = self.data[index], self.labels[index]

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return audio, target

    def __len__(self):
        return self.num_samples

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "vctk_info.txt"))

    def _read_info(self):
        info_path = os.path.join(
            self.root, self.processed_folder, "vctk_info.txt")
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.max_len = int(f.readline().split(",")[1])
            self.mean_len = float(f.readline().split(",")[1])
            self.std_len = float(f.readline().split(",")[1])
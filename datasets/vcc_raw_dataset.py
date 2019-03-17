import torch.utils.data as data
import numpy as np
import os
import os.path
import torch
import ast

class VCCRawDataset(data.Dataset):
    """`VCC2016 Dataset for <https://datashare.is.ed.ac.uk/handle/10283/2211>`.
    Based on torchaudio vctk.py <https://github.com/pytorch/audio>
    Args:
        root (string): Root directory of dataset where ``vcc2016_raw/processed/***.pt`` exist.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``torchaudio.transforms.Scale``
    """
    processed_folder = 'vcc2016_raw/processed'

    def __init__(self, root, eval=False, transform=None):
        self.root = os.path.expanduser(root)
        self.tensors = []
        self.labels = []
        self.num_samples = 0
        self.cached_pt = 0
        self.eval = eval
        self.transform = transform
        
        if self.eval:
            self.processed_folder = 'vcc2016_raw_eval/processed'

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self._read_info()

        self.tensors, self.labels = torch.load(os.path.join(
            self.root, self.processed_folder, "vcc2016_raw_train_{:04d}.pt".format(self.cached_pt)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (signal, speaker) where speaker is index of the target speaker id.
        """
        sig = self.tensors[index]
        speaker = self.labels[index]

        if self.transform is not None:
            sig = self.transform(sig)

        return sig, speaker

    def __len__(self):
        return self.num_samples

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "vcc2016_info.txt"))

    def _read_info(self):
        info_path = os.path.join(
            self.root, self.processed_folder, "vcc2016_info.txt")
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.ids = ast.literal_eval(f.readline().split(",", 1)[1])
            self.speaker_offset_idx = ast.literal_eval(f.readline().split(",", 1)[1])
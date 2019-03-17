import torch.utils.data as data
import numpy as np
import os
import os.path
import torch
import ast

class VCCWORLDDataset(data.Dataset):
    """`VCC2016 Dataset for <https://datashare.is.ed.ac.uk/handle/10283/2211>`.
    Based on torchaudio vctk.py <https://github.com/pytorch/audio>
    Preprocessed with WORLD vocoder.
    Args:
        root (string): Root directory of dataset where ``vcc2016/processed/***.pt`` exist.
        scale (boolean): Whether to scale spectral frames to between -1 and 1, or not.
    """
    processed_folder = 'vcc2016/processed'

    def __init__(self, root, scale=True, eval=False, load_conversion_data=False):
        self.root = os.path.expanduser(root)
        self.data = []
        self.spectra = []
        self.aperiodicity = []
        self.f0 = []
        self.energy = []
        self.labels = []
        self.num_samples = 0
        self.cached_pt = 0
        self.scale = scale
        self.eval = eval
        self.load_conversion_data = load_conversion_data
        
        if self.eval:
            self.processed_folder = 'vcc2016_eval/processed'

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self._read_info()
        self.current_chunk_idx = self.chunk_indices[self.cached_pt]
        self.spectra_scale = self.spectra_max - self.spectra_min

        self.spectra, self.labels = torch.load(os.path.join(
            self.root, self.processed_folder, "vcc2016_WORLD_train_{:04d}.pt".format(self.cached_pt)))
        if self.load_conversion_data:
            self.aperiodicity, self.f0, self.energy, _ = torch.load(os.path.join(
                self.root, self.processed_folder, "vcc2016_WORLD_conv_{:04d}.pt".format(self.cached_pt)))
    
    def get_data(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spectra, aperiodicity, f0, energy, speaker) where speaker is index of the target speaker id.
        """
        if index < self.current_chunk_idx[0] or index > self.current_chunk_idx[1]:
            for chunk, chunk_indices in self.chunk_indices.items():
                if index >= chunk_indices[0] and index <= chunk_indices[1]:
                    self.cached_pt = chunk
                    self.current_chunk_idx = chunk_indices
                    break

            self.spectra, self.labels = torch.load(os.path.join(
                self.root, self.processed_folder, "vcc2016_WORLD_train_{:04d}.pt".format(self.cached_pt)))
            if self.load_conversion_data:
                self.aperiodicity, self.f0, self.energy, _ = torch.load(os.path.join(
                    self.root, self.processed_folder, "vcc2016_WORLD_conv_{:04d}.pt".format(self.cached_pt)))

        index = index - self.current_chunk_idx[0]
        spectra = self.spectra[index]
        aperiodicity = self.aperiodicity[index]
        f0 = self.f0[index]
        energy = self.energy[index]
        speaker = self.labels[index]

        if self.scale:
            spectra = self.scale_spectra(spectra)

        return spectra, aperiodicity, f0, energy, speaker

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spectra, speaker) where speaker is index of the target speaker id.
        """
        if index < self.current_chunk_idx[0] or index > self.current_chunk_idx[1]:
            for chunk, chunk_indices in self.chunk_indices.items():
                if index >= chunk_indices[0] and index <= chunk_indices[1]:
                    self.cached_pt = chunk
                    self.current_chunk_idx = chunk_indices
                    break

            self.spectra, self.labels = torch.load(os.path.join(
                self.root, self.processed_folder, "vcc2016_WORLD_train_{:04d}.pt".format(self.cached_pt)))

        index = index - self.current_chunk_idx[0]
        spectra = self.spectra[index]
        speaker = self.labels[index]

        if self.scale:
            spectra = self.scale_spectra(spectra)

        return spectra, speaker
    
    def scale_spectra(self, spectra):
        """
        Scales spectra to (-1, +1) using the dataset's max and min values.
        """
        spectra = (spectra - self.spectra_min) / self.spectra_scale
        return torch.clamp(spectra, min=0., max=1.)*2. -1.

    def scale_spectra_back(self, spectra):
        """
        Scales spectra back to original scale.
        """
        return (spectra * 0.5 + 0.5) * self.spectra_scale + self.spectra_min
    
    def convert_f0(self, f0, src_id, target_id):
        """
        Convert fundamental frequency from source to target speaker
        """
        mu_src = self.speaker_mu[src_id]
        std_src = self.speaker_std[src_id]
        mu_target = self.speaker_mu[target_id]
        std_target = self.speaker_std[target_id]
        lf0 = torch.where(f0 > 1., torch.log(f0), f0)
        lf0 = torch.where(lf0 > 1., (lf0 - mu_src)/std_src * std_target + mu_target, lf0)
        return torch.where(lf0 > 1., torch.exp(lf0), lf0)

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
            self.chunk_indices = ast.literal_eval(f.readline().split(",", 1)[1])
            self.speaker_offset_idx = ast.literal_eval(f.readline().split(",", 1)[1])
            self.speaker_mu = ast.literal_eval(f.readline().split(",", 1)[1])
            self.speaker_std = ast.literal_eval(f.readline().split(",", 1)[1])
            self.spectra_min = torch.tensor(ast.literal_eval(f.readline().split(",", 1)[1]), dtype=torch.float32)
            self.spectra_max = torch.tensor(ast.literal_eval(f.readline().split(",", 1)[1]), dtype=torch.float32)
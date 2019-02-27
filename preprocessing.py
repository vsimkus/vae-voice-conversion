# Script to perform data preprocessing from wav files, including transformations

import torch
import vctk
import torchaudio.transforms as transforms
import argparse

parser = argparse.ArgumentParser(
        description='VCTK preprocessing helper script.')
parser.add_argument('--path', type=str, default='data', help='Path to the data folder that contains raw/VCTK-Corpus.zip file.')
args = parser.parse_args()

data = vctk.VCTK(args.path, download=True, downsample=True, dev_mode=True, transform=transforms.Compose([
    transforms.Scale(),
    transforms.PadTrim(max_len=308532),  # Set to length of the longest sample in the dataset

    # TODO decide on number of channels here
    transforms.MuLawEncoding(quantization_channels=256)
]))


data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1,
                                          shuffle=True)

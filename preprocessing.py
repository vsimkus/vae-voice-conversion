# Script to perform data preprocessing from wav files

import torch
import vctk
import argparse

parser = argparse.ArgumentParser(
        description='VCTK preprocessing helper script.')
parser.add_argument('--path', type=str, default='data', help='Path to the data folder that contains raw/VCTK-Corpus.zip file.')
args = parser.parse_args()

data = vctk.VCTK(args.path, download=True, downsample=True, dev_mode=True)

data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1,
                                          shuffle=True)

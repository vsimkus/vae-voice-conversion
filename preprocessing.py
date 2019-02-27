# Script to perform data preprocessing from wav files

import torch
import argparse
from vctk_preprocessor import VCTKPreprocessor

parser = argparse.ArgumentParser(
        description='VCTK preprocessing helper script.')
parser.add_argument('--path', type=str, default='data', help='Path to the data folder that contains raw/VCTK-Corpus.zip file.')
args = parser.parse_args()

data = VCTKPreprocessor(args.path, downsample=True, dev_mode=True)

data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1,
                                          shuffle=True)

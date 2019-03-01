# Script to perform data preprocessing from wav files

import torch
import argparse
from vctk_preprocessor import VCTKPreprocessor
from vctk_dataset import VCTKDataset
from arg_extractor import str2bool

parser = argparse.ArgumentParser(
        description='VCTK preprocessing helper script.')
parser.add_argument('--path', type=str, default='data', help='Path to the data folder that contains raw/VCTK-Corpus.zip file.')
parser.add_argument('--shuffle_order', type=str2bool, default=False, help='If true, shuffles the samples across the chunk-files.')
parser.add_argument('--trim_silence', type=str2bool, default=False, help='If true, trims silence from front and back ofthe audio.')
args = parser.parse_args()

data = VCTKPreprocessor(root=args.path, 
                        downsample=True,
                        trim_silence=args.trim_silence,
                        shuffle_order=args.shuffle_order,
                        dev_mode=True)

# Check we can load it.
dataset = VCTKDataset(root=args.path)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          shuffle=True)

# Script to perform data preprocessing from wav files
import torch
import argparse
from datasets.vctk_preprocessor import VCTKPreprocessor
from datasets.vcc_world_preprocessor import VCCWORLDPreprocessor
from datasets.vcc_raw_preprocessor import VCCRawPreprocessor
from arg_extractor import str2bool

parser = argparse.ArgumentParser(
        description='VCTK preprocessing helper script.')
parser.add_argument('--dataset', type=str, help='Dataset either VCCWORLD2016, VCCRaw2016, or VCTK.')
parser.add_argument('--path', type=str, default='data', help='Path to the data folder that contains raw/VCTK-Corpus.zip file.')
parser.add_argument('--shuffle_order', type=str2bool, default=False, help='If true, shuffles the samples across the chunk-files.')
parser.add_argument('--trim_silence', type=str2bool, default=False, help='If true, trims silence from front and back of the audio.')
parser.add_argument('--split_samples', type=str2bool, default=False, help='If true, splits samples to sample_length.')
parser.add_argument('--sample_length', type=int, default=8192, help='If split_samples is true, this is the sample length.')
parser.add_argument('--chunk_size', type=int, default=25000, help='Samples in a chunk-file.')
parser.add_argument('--extract_WORLD', type=str2bool, default=False, help='Whether to use WORLD vocoder for feature extraction.')
args = parser.parse_args()

if args.dataset == 'VCTK':
        data = VCTKPreprocessor(root=args.path,
                                downsample=True,
                                trim_silence=args.trim_silence,
                                shuffle_order=args.shuffle_order,
                                dev_mode=True,
                                chunk_size=args.chunk_size,
                                split_samples=args.split_samples,
                                sample_length=args.sample_length,
                                extract_WORLD=args.extract_WORLD)
elif args.dataset == 'VCCWORLD2016':
        data = VCCWORLDPreprocessor(root=args.path,
                                trim_silence=args.trim_silence,
                                dev_mode=True)
        data.process()
elif args.dataset == 'VCCRaw2016':
        data = VCCRawPreprocessor(root=args.path,
                                trim_silence=args.trim_silence,
                                dev_mode=True)
        data.process()
else:
        print('No such dataset!')
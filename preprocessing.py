# Script to perform data preprocessing from wav files, including transformations

import torch
import vctk
import torchaudio.transforms as transforms

zip_path = ''  # Provide local path to VCTK zip file
data = vctk.VCTK(zip_path, download=True, downsample=True, dev_mode=True, transform=transforms.Compose([
    transforms.Scale(),
    transforms.PadTrim(max_len=16000),  # TODO decide on max length

    # TODO decide on number of channels here
    transforms.MuLawEncoding(quantization_channels=256)
]))


data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=1,
                                          shuffle=True)

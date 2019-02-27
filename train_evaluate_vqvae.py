yimport os

import torch
import numpy as np
import torchaudio_transforms as transforms
import os
from arg_extractor import get_args
from experiment_builder import VQVAEExperimentBuilder
from model_architectures import VQVAE
from vctk_dataset import VCTKDataset

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

data_dir = os.environ['DATASET_DIR']

# Load dataset
dataset = VCTKDataset(root=data_dir, transform=transforms.Compose([
    transforms.PadTrim(max_len=args.input_max_len),
    # TODO decide on number of channels here
    # transforms.MuLawEncoding(quantization_channels=256)
]))

# Split train and test/validation sets
train_length = int(0.9*len(dataset)) # 90% for training, and 10% for test/validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths=[train_length, len(dataset)-train_length])

# Create data loaders
train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_data = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_data = val_data

vqvae_model = VQVAE(
    input_shape=(args.batch_size, 1, args.input_max_len),
    encoder_arch=args.encoder,
    vq_arch=args.vq,
    generator_arch=args.generator,
    num_speakers=109)

vqvae_experiment = VQVAEExperimentBuilder(network_model=vqvae_model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    learning_rate=args.learning_rate,
                                    commit_coefficient=args.commit_coefficient,
                                    gpu_id=args.gpu_id,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data,
                                    val_data=val_data,
                                    test_data=test_data)

experiment_metrics, test_metrics = vqvae_experiment.run_experiment()  # run experiment and return experiment metrics

import numpy as np
from arg_extractor import get_args

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torch
import torchaudio_transforms as transforms
from experiment_builder import VAEExperimentBuilder
from model_architectures import VAE
from vctk_dataset import VCTKDataset
from samplers import ChunkEfficientRandomSampler

torch.manual_seed(seed=args.seed) # sets pytorch's seed

# Load dataset
dataset_path = args.dataset_root_path
dataset = VCTKDataset(root=dataset_path, transform=transforms.Compose([
    transforms.PadTrim(max_len=args.input_max_len)
]))

if args.tuning_mode:
    # Split train and test/validation sets
    train_length = int(0.25*len(dataset)) # 25% for training,
    val_length = int(0.05*len(dataset)) # 5% for test/validation, and discard the rest
    print('Running in tuning mode, with train set size {} and val size {}'.format(train_length, val_length))
    train_dataset = torch.utils.data.Subset(dataset, range(train_length))
    val_dataset = torch.utils.data.Subset(dataset, range(train_length, train_length+val_length))
else:
    # Split train and test/validation sets
    train_length = int(0.9*len(dataset)) # 90% for training, and 10% for test/validation
    train_dataset = torch.utils.data.Subset(dataset, range(train_length))
    val_dataset = torch.utils.data.Subset(dataset, range(train_length, len(dataset)))

# Create data loaders
chunk_size = 1500 # This is the number of samples in a chunkfile
train_data = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=1,
                                        sampler=ChunkEfficientRandomSampler(train_dataset,
                                                                            chunk_size)
                                        )
val_data = torch.utils.data.DataLoader(val_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=1,
                                        sampler=ChunkEfficientRandomSampler(val_dataset,
                                                                            chunk_size,
                                                                            first_chunk_offset=train_length % chunk_size
                                                                            )
                                        )
test_data = val_data

vae_model = VAE(
    input_shape=(1, 1, args.input_max_len),
    encoder_arch=args.encoder,
    num_latents=args.num_latents,
    generator_arch=args.generator,
    num_speakers=109)

vae_experiment = VAEExperimentBuilder(network_model=vae_model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    learning_rate=args.learning_rate,
                                    commit_coefficient=args.commit_coefficient,
                                    device=device,
                                    continue_from_epoch=args.continue_from_epoch,
                                    print_timings=args.print_timings,
                                    train_data=train_data,
                                    val_data=val_data,
                                    test_data=test_data)

experiment_metrics, test_metrics = vae_experiment.run_experiment()  # run experiment and return experiment metrics

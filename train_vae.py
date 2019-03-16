import numpy as np
from arg_extractor import get_args

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torch
import torchaudio_transforms as transforms
from experiment_builders.vae_builder import VAEWORLDExperimentBuilder
from models.vae import VAE
from datasets.vcc_world_dataset import VCCWORLDDataset
from samplers import ChunkEfficientRandomSampler

torch.manual_seed(seed=args.seed)

if args.dataset == 'VCCWORLD2016':
    print('VCC2016 dataset WORLD features.')

    dataset_path = args.dataset_root_path
    train_dataset = VCCWORLDDataset(root=dataset_path, scale=True)
    chunk_size = train_dataset.chunk_size # This is the number of samples in a chunkfile
    val_dataset = VCCWORLDDataset(root=dataset_path, scale=True, eval=True)
    val_chunk_size = val_dataset.chunk_size # This is the number of samples in a chunkfile

    # Create data loaders
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
                                                                                val_chunk_size)
                                            )
else:
    raise Exception('No such dataset!')

vae_model = VAE(
    input_shape=(1, 1, args.input_len),
    encoder_arch=args.encoder,
    generator_arch=args.generator,
    latent_dim=args.latent_dim,
    num_speakers=args.num_speakers,
    speaker_dim=args.speaker_dim,
    use_gated_convolutions=args.use_gated_convolutions)


vae_experiment = VAEWORLDExperimentBuilder(network_model=vae_model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    learning_rate=args.learning_rate,
                                    device=device,
                                    continue_from_epoch=args.continue_from_epoch,
                                    print_timings=args.print_timings,
                                    train_data=train_data,
                                    val_data=val_data)

experiment_metrics, test_metrics = vae_experiment.run_experiment()
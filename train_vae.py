import numpy as np
from util.arg_extractor import get_args

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torch
import util.torchaudio_transforms as transforms
from experiment_builders.vae_builder import VAEWORLDExperimentBuilder
from experiment_builders.vae_builder import VAERawExperimentBuilder
from models.vae import VAE
from models.common_networks import QuantisedInputModuleWrapper
from datasets.vcc_world_dataset import VCCWORLDDataset
from datasets.vcc_raw_dataset import VCCRawDataset
from util.samplers import ChunkEfficientRandomSampler

torch.manual_seed(seed=args.seed)

vae_model = VAE(
    input_shape=(1, 1, args.input_len),
    encoder_arch=args.encoder,
    generator_arch=args.generator,
    latent_dim=args.latent_dim,
    num_speakers=args.num_speakers,
    speaker_dim=args.speaker_dim,
    use_gated_convolutions=args.use_gated_convolutions)


if args.dataset == 'VCCWORLD2016':
    print('VCC2016 dataset WORLD features.')

    dataset_path = args.dataset_root_path
    train_dataset = VCCWORLDDataset(root=dataset_path, scale=True)
    val_dataset = VCCWORLDDataset(root=dataset_path, scale=True, eval=True)

    # Create data loaders
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        sampler=ChunkEfficientRandomSampler(train_dataset, train_dataset.chunk_indices))
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        sampler=ChunkEfficientRandomSampler(val_dataset, train_dataset.chunk_indices))
    
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

elif args.dataset == 'VCCRaw2016':
    print('VCC2016 dataset raw features.')

    dataset_path = args.dataset_root_path
    train_dataset = VCCRawDataset(root=dataset_path, transform=transforms.Compose([
        transforms.MuLawEncoding(quantization_channels=args.num_input_quantization_channels)
    ]))
    val_dataset = VCCRawDataset(root=dataset_path, eval=True, transform=transforms.Compose([
        transforms.MuLawEncoding(quantization_channels=args.num_input_quantization_channels)
    ]))

    # Create data loaders
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1)
    
    quantised_input_vae_model = QuantisedInputModuleWrapper(args.num_input_quantization_channels, vae_model)
    
    vae_experiment = VAERawExperimentBuilder(network_model=quantised_input_vae_model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    learning_rate=args.learning_rate,
                                    device=device,
                                    continue_from_epoch=args.continue_from_epoch,
                                    print_timings=args.print_timings,
                                    train_data=train_data,
                                    val_data=val_data)

else:
    raise Exception('No such dataset!')

experiment_metrics, test_metrics = vae_experiment.run_experiment()
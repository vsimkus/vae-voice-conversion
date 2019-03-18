import numpy as np
from util.arg_extractor import get_args

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)

import torch
import math
import operator
import util.torchaudio_transforms as transforms
from experiment_builders.vqvae_builder import VQVAEWORLDExperimentBuilder, VQVAERawExperimentBuilder
from models.vqvae import VQVAE
from models.common_networks import QuantisedInputModuleWrapper
from datasets.vcc_world_dataset import VCCWORLDDataset
from datasets.vcc_raw_dataset import VCCRawDataset
from datasets.vctk_dataset import VCTKDataset
from util.samplers import ChunkEfficientRandomSampler

torch.manual_seed(seed=args.seed)

vqvae_model = VQVAE(
    input_shape=(1, 1, args.input_len),
    encoder_arch=args.encoder,
    vq_arch=args.vq,
    generator_arch=args.generator,
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
        sampler=ChunkEfficientRandomSampler(val_dataset, val_dataset.chunk_indices))
    
    vqvae_experiment = VQVAEWORLDExperimentBuilder(network_model=vqvae_model,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        commit_coefficient=args.commit_coefficient,
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
    
    quantised_input_vqvae_model = QuantisedInputModuleWrapper(args.num_input_quantization_channels, vqvae_model)
    
    vqvae_experiment = VQVAERawExperimentBuilder(network_model=quantised_input_vqvae_model,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        commit_coefficient=args.commit_coefficient,
                                        learning_rate=args.learning_rate,
                                        device=device,
                                        continue_from_epoch=args.continue_from_epoch,
                                        print_timings=args.print_timings,
                                        train_data=train_data,
                                        val_data=val_data)

elif args.dataset == 'VCTKRaw':
    print('VCTK dataset raw features.')

    dataset_path = args.dataset_root_path
    chunk_size = 10000 # Number of files in the chunkfile
    dataset = VCTKDataset(root=dataset_path, chunk_size=chunk_size, transform=transforms.Compose([
        transforms.MuLawEncoding(quantization_channels=args.num_input_quantization_channels)
    ]))
    
    num_chunks = math.ceil(len(dataset) / chunk_size)
    chunk_indices = {i:(i*chunk_size, (i+1)*chunk_size-1) \
                        if i < num_chunks-1 \
                        else (i*chunk_size, len(dataset)) \
                        for i in range(num_chunks) }
    
    # Last two chunks are for validation.
    train_chunk_indices = {key:value for key, value in chunk_indices.items() if key < num_chunks-2}
    val_chunk_indices = {key:value  for key, value in chunk_indices.items() if key >= num_chunks-2}

    train_dataset = torch.utils.data.Subset(dataset, range(min(train_chunk_indices.items(), key=operator.itemgetter(0))[1][0], 
                                                            max(train_chunk_indices.items(), key=operator.itemgetter(0))[1][1]+1))
    val_dataset = torch.utils.data.Subset(dataset, range(min(val_chunk_indices.items(), key=operator.itemgetter(0))[1][0], 
                                                            max(val_chunk_indices.items(), key=operator.itemgetter(0))[1][1]+1))

    # Re-compute chunk indices for the validation subset
    val_num_chunks = math.ceil(len(val_dataset) / chunk_size)
    val_chunk_indices = {i:(i*chunk_size, (i+1)*chunk_size-1) \
                        if i < val_num_chunks-1 \
                        else (i*chunk_size, len(val_dataset)) \
                        for i in range(val_num_chunks) }

    # Create data loaders
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        sampler=ChunkEfficientRandomSampler(train_dataset, train_chunk_indices))
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        sampler=ChunkEfficientRandomSampler(val_dataset, val_chunk_indices))
    
    quantised_input_vqvae_model = QuantisedInputModuleWrapper(args.num_input_quantization_channels, vqvae_model)
    
    vqvae_experiment = VQVAERawExperimentBuilder(network_model=quantised_input_vqvae_model,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        commit_coefficient=args.commit_coefficient,
                                        learning_rate=args.learning_rate,
                                        device=device,
                                        continue_from_epoch=args.continue_from_epoch,
                                        print_timings=args.print_timings,
                                        train_data=train_data,
                                        val_data=val_data)

else:
    raise Exception('No such dataset!')


experiment_metrics, test_metrics = vqvae_experiment.run_experiment()
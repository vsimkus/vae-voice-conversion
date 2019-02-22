import torch
import numpy as np
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from model_architectures import VQVAE

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

train_data = data_providers.VCTKDataProvider('train', batch_size=args.batch_size, rng=rng)
val_data = data_providers.VCTKDataProvider('valid', batch_size=args.batch_size, rng=rng)
test_data = data_providers.VCTKDataProvider('test', batch_size=args.batch_size,rng=rng)

vqvae_model = VQVAE(
    input_shape=(args.batch_size, 1, 1000, #TODO: set width, I think this can be arbitraty though, since no filter is fixed to the size of our samples.
    encoder_arch=args.encoder,
    vq_arch=args.vq,
    generator_arch=args.generator,
    num_speakers=109) #TODO: set from dataset

# TODO: update this once Experiment builder is done.
vqvae_experiment = ExperimentBuilder(network_model=vqvae_model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    gpu_id=args.gpu_id, 
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, 
                                    val_data=val_data,
                                    test_data=test_data)

experiment_metrics, test_metrics = vqvae_experiment.run_experiment()  # run experiment and return experiment metrics
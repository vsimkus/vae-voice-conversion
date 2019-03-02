import torch
import torchaudio
import os.path
from arg_extractor import get_args
from experiment_builder import VQVAEExperimentBuilder
from model_architectures import VQVAE
from vctk_preprocessor import read_audio

args = get_args() 

assert args.continue_from_epoch == -2 or args.continue_from_epoch > 1, \
    'This script is for evaluation only, please specify the epoch to run, or -2 for latest.'

# Load audio
audio_path = os.path.expanduser(args.eval_audio_path)
torchaudio.initialize_sox()
audio, sr = read_audio(audio_path, downsample=True, trim_silence=True)
torchaudio.shutdown_sox()

# Prepare speaker tensor
speaker_id = torch.tensor(args.eval_speaker_id, dtype=torch.long)

# Prepare model
vqvae_model = VQVAE(
    input_shape=(1, 1, args.input_max_len),
    encoder_arch=args.encoder,
    vq_arch=args.vq,
    generator_arch=args.generator,
    num_speakers=109)

# Load model from state
vqvae_experiment = VQVAEExperimentBuilder(network_model=vqvae_model,
                                    experiment_name=args.experiment_name,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    learning_rate=args.learning_rate,
                                    commit_coefficient=args.commit_coefficient,
                                    gpu_id=args.gpu_id,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=None,
                                    val_data=None,
                                    test_data=None,
                                    num_epochs=None)

audio = audio.unsqueeze(0)
out = vqvae_experiment.convert(x=audio, y=speaker_id)[0]
# print(audio)
# import numpy as np
# print(out.detach().numpy() < 0)


#TODO: Upsampling?

# Check out path
out_path = os.path.expanduser(args.eval_out_path)
if not os.path.exists(out_path):
    os.mkdir(out_path)

out_filename = '{}_{}.wav'.format(os.path.basename(audio_path).split('.')[0], args.eval_speaker_id)
out_file_path = os.path.join(out_path, out_filename)

# Save as audio
torchaudio.save(filepath=out_file_path, src=out, sample_rate=sr)
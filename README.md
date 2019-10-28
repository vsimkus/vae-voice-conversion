# Voice Conversion on unaligned data

Voice Conversion (VC) is widely desirable across many industries and applications, including speaker anonymisation, film dubbing, gaming, and voice restoration for people who have lost their ability to speak. In this work we compare standard VAE, VQ-VAE and Gumbel VAE models as approaches to VC on the Voice Conversion Challenge 2016 dataset.
We assess speech reconstruction and VC performance on both spectral frames as obtained from a WORLD vocoder and on the raw waveform data.

The full report and evaluation results can be found [here](report/report.pdf).

## How to train your VC model

### 1. Preprocess data

Place the raw [VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211) dataset in `data/vcc2016_raw/vcc2016_training.zip` (raw audio features) or `data/vcc2016/vcc2016_training.zip` (WORLD features), or [VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651) dataset in `data/vctk/VCTK-Corpus.zip`.

To generate the preprocessed data files run one of the following:

* VCC2016 Raw data

    ```bash
    python preprocessing.py --dataset=VCCRaw2016 --trim_silence=True
    ```

* VCC2016 WORLD data

    ```bash
    python preprocessing.py --dataset=VCCWORLD2016 --trim_silence=True
    ```

* VCTK Raw data

    ```bash
    python preprocessing.py --dataset=VCTK --trim_silence=True --shuffle_order=True --split_samples=True
    ```

### 2. Train model

Run the following on a compute cluster to train the model.

```bash
python train_vqvae.py \ # Or train_vae.py, train_joint_vae.py
            --use_gpu=True \ # Whether to use GPU
            --gpu_id='0,1' \ # GPU ids to use
            --filepath_to_arguments_json_file="experiment_configs/config_file.json" \ # Model and experiment configuration file
            --dataset_root_path='data'
```

## How to evaluate or perform VC

In order to evaluate the model you first have to create and load the trained model. Then you have to prepare your audio data using WORLD or mu-law preprocessing, as well as padding/trimming such that the input to the model is of the correct length. After conversion, you have to postprocess to produce the audio file. An example is given in this section.

### 1. Load configuration

```python
from util.arg_extractor import extract_args_from_json

args = extract_args_from_json('experiment_configs/config_file.json')
```

### 2. Create model

#### VQVAE

```python
from models.vqvae import VQVAE

model = VQVAE(
    input_shape=(1, 1, args.input_len),
    encoder_arch=args.encoder,
    vq_arch=args.vq,
    generator_arch=args.generator,
    num_speakers=args.num_speakers,
    speaker_dim=args.speaker_dim,
    use_gated_convolutions=args.use_gated_convolutions)
```

#### VAE

```python
from models.vae import VAE

model = VAE(
    input_shape=(1, 1, args.input_len),
    encoder_arch=args.encoder,
    generator_arch=args.generator,
    latent_dim=args.latent_dim,
    num_speakers=args.num_speakers,
    speaker_dim=args.speaker_dim,
    use_gated_convolutions=args.use_gated_convolutions)
```

#### JointVAE

```python
from models.joint_vae import JointVAE

model = JointVAE(
    input_shape=(1, 1, args.input_len),
    encoder_arch=args.encoder,
    generator_arch=args.generator,
    latent_dim=args.latent_dim,
    num_latents=args.num_latents,
    temperature=args.temperature,
    num_speakers=args.num_speakers,
    speaker_dim=args.speaker_dim,
    use_gated_convolutions=args.use_gated_convolutions)
```

### 3. Load model weights

To load the model weights we us the same experiment builders as used in training.

```python
from experiment_builders.vqvae_builder import VQVAERawExperimentBuilder

# To load the model weights use VQVAERawExperimentBuilder, VQVAEWORLDExperimentBuilder, VAERawExperimentBuilder, VAEWORLDExperimentBuilder, JointVAERawExperimentBuilder, or JointVAEWORLDExperimentBuilder depending on the experiment
builder = VQVAERawExperimentBuilder(network_model=model,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    learning_rate=args.learning_rate,
                                    commit_coefficient=args.commit_coefficient, # This argument is only needed in VQVAE experiment builders
                                    device=torch.device('cpu'),
                                    continue_from_epoch=epoch, # Epoch of the model to load (should be your best validation model)
                                    train_data=None,
                                    val_data=None)
```

### 4. Perform conversion

#### Raw data feature experiments

```python
import torchaudio
import util.torchaudio_transforms as transforms
from datasets.vcc_preprocessor import read_audio # Or import from vctk_preprocessor respectively

# Prepare mu-law encoding transformers
mulaw = transforms.MuLawEncoding(quantization_channels=args.num_input_quantization_channels)
mulaw_expanding = transforms.MuLawExpanding(quantization_channels=args.num_input_quantization_channels)

# Load audio
audio_path = os.path.expanduser(audio_path)
torchaudio.initialize_sox()
audio, sr = read_audio(audio_path, trim_silence=True)
torchaudio.shutdown_sox()

# Prepare an audio piece of appropriate length, e.g. as follows
audio = audio.unsqueeze(0)
audio_len = audio.shape[-1]
padding = transforms.PadTrim(math.ceil(audio.shape[-1] / args.input_len) * args.input_len)
audio = padding(audio.squeeze(0)).unsqueeze(0)
audio_split = audio.view(int(audio.shape[-1] / args.input_len), 1, args.input_len)

# Set target speaker id
target_speaker_id = torch.tensor(target_speaker_id, dtype=torch.long)

# Voice conversion
out_mulaw = builder.convert(x=mulaw(audio_split), y=target_speaker_id)

# Postprocess
out = mulaw_expanding(out_mulaw).detach().view(1, -1)
out = out[:, :audio_len]

# Save as audio file
torchaudio.save(filepath=out_file_path, src=out, sample_rate=sr)
```

#### WORLD feature experiments

```python
from data.vcc_world_dataset import VCCWORLDDataset
from datasets.vcc_world_preprocessor import read_audio_and_extract_features, synthesize_from_WORLD_features

# Load audio
audio_path = os.path.expanduser(audio_path)
spectra, aperiodicity, f0, energy = read_audio_and_extract_features(audio_path)

# Set target speaker id
target_speaker_id = torch.tensor(target_speaker_id, dtype=torch.long)

# Voice conversion
dataset = VCCWORLDDataset('data', scale=True)
spectra_scaled = dataset.scale_spectra(torch.tensor(spectra)).unsqueeze(1)
spectra_out = builder.convert(x=spectra_scaled, y=speaker_id)
spectra_out = dataset.scale_spectra_back(spectra_out)
f0_converted = dataset.convert_f0(torch.tensor(f0), source_speaker_id, args.eval_speaker_id)
spectra_out = spectra_out.squeeze(1)
# Synthesize audio
audio_out = synthesize_from_WORLD_features(f0_converted.numpy(), spectra_out.numpy(), aperiodicity, energy)
audio_out = np.clip(audio_out, a_min=-0.9, a_max=0.9)

# Save as audio
torchaudio.save(filepath=out_file_path, src=torch.tensor(audio_out.copy()), sample_rate=16000)
```

## Models

In our evaluation we have investigated three different VAE models.

* [VAE](https://arxiv.org/abs/1610.04019), implemented in [models/vae.py](models/vae.py).
* [VQVAE](https://papers.nips.cc/paper/7210-neural-discrete-representation-learning), implemented in [models/vqvae.py](models/vqvae.py) and [models/vq_functions.py](models/vq_functions.py).
* [JointVAE](https://arxiv.org/abs/1804.00104), implemented in [models/joint_vae.py](models/joint_vae.py).

## Software dependencies

* PyTorch v1.0.0 or later
* numpy
* pillow
* tqdm
* [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) (for extracting WORLD features)
* [torchaudio](https://github.com/pytorch/audio) (for preprocessing raw audio)

## VCTK dataset modifications

The VCTK dataset has some silent files, hence the following audio samples were removed

* `p323_424`, `p306_151`, `p351_361`, `p345_292`, `p341_101`, `p306_352`.

## Contributors

* Vaidotas Simkus
* Simon Valentin
* Will Greedy

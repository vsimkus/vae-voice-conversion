import torch.utils.data as data
import numpy as np
import os
import os.path
import shutil
import errno
import torch
import torchaudio
import torch.nn.functional as F
from random import shuffle
import pyworld as pw

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def make_manifest(dir):
    audios = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = path
                    audios.append(item)

    return audios

def load_ids(dir):
    """Create a dictionary to convert VCC2016 participant ID to integer ID."""
    ids = dict()
    id = 0
    dir = os.path.expanduser(dir)
    for speaker in sorted(os.listdir(dir)):
        d = os.path.join(dir, speaker)
        if not os.path.isdir(d):
            continue

        fname_no_ext = os.path.basename(d).rsplit(".", 1)[0]
        ids[fname_no_ext] = id
        id += 1

    return ids

def read_audio(fp, trim_silence=False):
    if trim_silence:
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fp)

        E.append_effect_to_chain("silence", [1, 100, 1])
        E.append_effect_to_chain("reverse")
        E.append_effect_to_chain("silence", [1, 100, 1])
        E.append_effect_to_chain("reverse")
    
        sig, sample_rate = E.sox_build_flow_effects()
    else:
        sig, sample_rate = torchaudio.load(fp)
    sig = sig.contiguous()
    return sig, sample_rate

def raw2WORLDfeatures(signal, fs=16000, fft_size=1024):
    _f0, t = pw.dio(signal, fs, f0_ceil=500) # raw pitch contour extractor
    f0 = pw.stonemask(signal, _f0, t, fs)            # pitch refinement
    spectra = pw.cheaptrick(signal, f0, t, fs, fft_size=fft_size)
    aperiodicity = pw.d4c(signal, f0, t, fs, fft_size=fft_size) # extract aperiodicity
    return f0, spectra, aperiodicity

def read_audio_and_extract_features(fp, trim_silence=False):
    signal, sample_rate = read_audio(fp, trim_silence)

    f0, spectra, aperiodicity = raw2WORLDfeatures(signal.numpy()[0].astype(np.float64), fs=sample_rate)
    energy = np.sum(spectra + 1e-10, axis=1, keepdims=True)
    # Normalize spectra to unit sum
    spectra = np.log10(spectra / energy)
    spectra = spectra.astype(np.float32)
    aperiodicity = aperiodicity.astype(np.float32)
    f0 = f0.astype(np.float32)
    energy = energy.astype(np.float32)
    return spectra, aperiodicity, f0, energy

def synthesize_from_WORLD_features(f0, spectra, aperiodicity, energy, sample_rate=16000):
    if isinstance(spectra, list):
        spectra = np.vstack(spectra)
    if isinstance(aperiodicity, list):
        aperiodicity = np.vstack(aperiodicity)
    if isinstance(energy, list):
        energy = np.vstack(energy)
    if isinstance(f0, list):
        f0 = np.array(f0)

    sp = np.power(10., spectra)
    sp = energy * sp
    return pw.synthesize(
            f0.astype(np.float64).copy(order='C'),
            sp.astype(np.float64).copy(order='C'),
            aperiodicity.astype(np.float64).copy(order='C'),
            sample_rate
        )


class VCCWORLDPreprocessor(): # TODO: refactor
    """`VCC2016 Preprocessor for <https://datashare.is.ed.ac.uk/handle/10283/2211>`.
    Based on torchaudio vctk.py <https://github.com/pytorch/audio>
    Extracts WORLD features.
    Args:
        root (string): Root directory of dataset where the dataset should be stored in vcc2016/raw/, vcc2016/processed/ directories.
        trim_silence (bool, optional): if true, trim trailing silence in from and end of the samples. (default=False)
        shuffle_order (bool, optional): if true, shuffle the audio files across the chunk-files. (default=False)
        dev_mode(bool, optional): if true, clean up is not performed on raw files.  Useful to keep raw audio and transcriptions.
    """
    raw_folder = 'vcc2016/raw'
    processed_folder = 'vcc2016/processed'
    zip_path = 'vcc2016_training.zip'  # path to local zip file
    dset_path = 'vcc2016_training'

    def __init__(self, root, trim_silence=False, dev_mode=True):
        self.root = os.path.expanduser(root)
        self.trim_silence = trim_silence
        self.dev_mode = dev_mode
        self.labels = []
        self.num_samples = 0
        self.max_len = 0
        self.mean_len = 0.
        self.std_len = 0.

        if self.trim_silence:
            print('Will trim trailing silence.')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "vcc2016_info.txt"))

    def _write_info(self, num_items):
        info_path = os.path.join(
            self.root, self.processed_folder, "vcc2016_info.txt")
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("ids,{}\n".format(self.ids))
            f.write("chunk_indices,{}\n".format(self.chunk_indices))
            f.write("speaker_offset_idx,{}\n".format(self.speaker_offset_idx))

    def process(self):
        """Process the VCC2016 data if it doesn't exist in processed_folder already."""
        import zipfile

        if self._check_exists():
            return

        raw_abs_dir = os.path.join(self.root, self.raw_folder)
        processed_abs_dir = os.path.join(self.root, self.processed_folder)
        dset_abs_path = os.path.join(
            self.root, self.raw_folder, self.dset_path)

        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        zip_path = self.zip_path
        print('Unzipping', zip_path)
        filename = zip_path.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.isfile(file_path):
            shutil.copy2(zip_path, file_path)

        if not os.path.exists(dset_abs_path):
            with zipfile.ZipFile(file_path) as zip_f:
                zip_f.extractall(raw_abs_dir)
        else:
            print("Using existing raw folder")
        if not self.dev_mode:
            os.unlink(file_path)

        # process and save as torch files
        torchaudio.initialize_sox()
        print('Processing...')
        shutil.copyfile(
            os.path.join(dset_abs_path, "README"),
            os.path.join(processed_abs_dir, "VCC2016_README")
        )
        audios = make_manifest(dset_abs_path)
        self.ids = load_ids(dset_abs_path)
        print("Found {} audio files".format(len(audios)))
        
        print('Extracting WORLD features.')
        spectras = []
        aperiodicities = []
        f0s = []
        energies = []
        labels = []
        chunk_id = 0
        samples = 0
        self.speaker_offset_idx = {}
        self.chunk_indices = {}
        current_chunk_start_idx = 0
        prev_speaker = -1
        for f in audios:
            speaker = f.split("/", -1)[-2]
            spectra, aperiodicity, f0_, energy = read_audio_and_extract_features(f, trim_silence=self.trim_silence)

            # New speaker, save current chunk and start a fresh chunk
            if prev_speaker != -1 and speaker != prev_speaker:
                self.speaker_offset_idx[self.ids[speaker]] = samples
                print('Speaker {}: start idx: {}'.format(speaker, samples))
                self.chunk_indices[chunk_id] = (current_chunk_start_idx, samples-1)
                prev_speaker = speaker
                current_chunk_start_idx = samples

                self.save_WORLD_chunk(chunk_id, spectras, aperiodicities, f0s, energies, labels)
                chunk_id += 1
                spectras = []
                aperiodicities = []
                f0s = []
                energies = []
                labels = []
            elif prev_speaker == -1:
                prev_speaker = speaker

            # Add each spectral frame as a separate datapoint
            for i in range(spectra.shape[0]):
                sp = torch.tensor(spectra[i]).unsqueeze(0).float()
                ap = torch.tensor(aperiodicity[i]).unsqueeze(0).float()
                f0 = torch.tensor(f0_[i]).float()
                en = torch.tensor(energy[i]).float()

                spectras.append(sp)
                aperiodicities.append(ap)
                f0s.append(f0)
                energies.append(en)
                labels.append(self.ids[speaker])
                
                samples += 1
                    

        if len(spectras) > 0 :
            self.chunk_indices[chunk_id] = (current_chunk_start_idx, samples-1)
            self.save_WORLD_chunk(chunk_id, spectras, aperiodicities, f0s, energies, labels)
        
        self._write_info(samples)

        if not self.dev_mode:
            shutil.rmtree(raw_abs_dir, ignore_errors=True)
        torchaudio.shutdown_sox()
        print('Done!')

    def save_WORLD_chunk(self, chunk_id, spectra, aperiodicity, f0, energies, labels):
        print('Saving chunk {} with speakers {}'.format(chunk_id, set(labels)))
        data_training = (spectra, labels)
        torch.save(
            data_training,
            os.path.join(
                self.root,
                self.processed_folder,
                "vcc2016_WORLD_train_{:04d}.pt".format(chunk_id)
            )
        )
        data_conversion = (aperiodicity, f0, energies, labels)
        torch.save(
            data_conversion,
            os.path.join(
                self.root,
                self.processed_folder,
                "vcc2016_WORLD_conv_{:04d}.pt".format(chunk_id)
            )
        )
    
    def extract_dataset_max_min_and_speaker_profiles(self): 
        # TODO: refactor for this to work with split conversion/spectra files!
        processed_abs_dir = os.path.join(self.root, self.processed_folder)

        chunk_files = make_chunk_manifest(processed_abs_dir)
        print(chunk_files)

        current_speaker = -1
        speaker_f0 = []
        speaker_mu = {}
        speaker_std = {}
        spectra_all = []
        for chunk_file in chunk_files:
            spectra, _, f0, _, speaker = torch.load(chunk_file)

            for sp, f0_, speaker_id in zip(spectra, f0, speaker):
                if speaker_id != current_speaker and current_speaker != -1:
                    assert current_speaker not in speaker_mu.keys(), \
                        'We need the speaker samples to be sequential to compute the speaker profiles!'
                    mu, std = extract_speaker_logf0_mean_and_variance(speaker_f0)

                    speaker_mu[current_speaker] = mu
                    speaker_std[current_speaker] = std

                    speaker_f0 = []

                current_speaker = speaker_id
                speaker_f0.append(f0_.item())
                spectra_all.append(sp)

        # Compute profile of last speaker
        if len(speaker_mu) > 0 and current_speaker != -1:
            assert current_speaker not in speaker_mu.keys(), \
                'We need the speaker samples to be sequential to compute the speaker profiles!'
            mu, std = extract_speaker_logf0_mean_and_variance(speaker_f0)

            speaker_mu[current_speaker] = mu
            speaker_std[current_speaker] = std
            speaker_f0 = []
        
        # Append speaker logf0 mu and std
        info_path = os.path.join(
            self.root, self.processed_folder, "vcc2016_info.txt")
        with open(info_path, "a") as f:
            f.write("speaker_mu,{}\n".format(speaker_mu))
            f.write("speaker_std,{}\n".format(speaker_std))

        # Compute spectra max and min values
        spectra_all = np.concatenate(spectra_all, axis=0)
        q005 = np.percentile(spectra_all, 0.5, axis=0)
        q995 = np.percentile(spectra_all, 99.5, axis=0)
        # TODO: use json
        with open(info_path, "a") as f:
            q005_out = ','.join(map(str, q005))
            q005_out = '[{}]'.format(q005_out)
            f.write("spectra_min,{}\n".format(q005_out))

        with open(info_path, "a") as f:
            q995_out = ','.join(map(str, q995))
            q995_out = '[{}]'.format(q995_out)
            f.write("spectra_max,{}\n".format(q995_out))


def extract_speaker_logf0_mean_and_variance(speaker_f0):
    speaker_f0 = np.array(speaker_f0)
    print(speaker_f0.shape)
    speaker_f0 = speaker_f0[speaker_f0 > 2]
    speaker_f0 = np.log(speaker_f0)
    return speaker_f0.mean(), speaker_f0.std()

def make_chunk_manifest(dir):
    chunks = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.pt'):
                path = os.path.join(root, fname)
                item = path
                chunks.append(item)

    return sorted(chunks)
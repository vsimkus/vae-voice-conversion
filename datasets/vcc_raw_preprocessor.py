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


class VCCRawPreprocessor(): # TODO: refactor
    """`VCC2016 Preprocessor for <https://datashare.is.ed.ac.uk/handle/10283/2211>`.
    Based on torchaudio vctk.py <https://github.com/pytorch/audio>
    Args:
        root (string): Root directory of dataset where the dataset should be stored in vcc2016/raw/, vcc2016/processed/ directories.
        trim_silence (bool, optional): if true, trim trailing silence in from and end of the samples. (default=False)
        shuffle_order (bool, optional): if true, shuffle the audio files across the chunk-files. (default=False)
        dev_mode(bool, optional): if true, clean up is not performed on raw files.  Useful to keep raw audio and transcriptions.
        sample_length(int): splits samples into multiple samples of this length. (default=8192)
    """
    raw_folder = 'vcc2016_raw/raw'
    processed_folder = 'vcc2016_raw/processed'
    zip_path = 'vcc2016_training.zip'  # path to local zip file
    dset_path = 'vcc2016_training'

    def __init__(self, root, trim_silence=False, dev_mode=True, sample_length=8192):
        self.root = os.path.expanduser(root)
        self.trim_silence = trim_silence
        self.dev_mode = dev_mode
        self.labels = []
        self.num_samples = 0
        self.max_len = 0
        self.mean_len = 0.
        self.std_len = 0.
        self.sample_length = sample_length

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
        
        print('Splitting samples to length {}'.format(self.sample_length))
        tensors = []
        labels = []
        chunk_id = 0
        samples = 0
        self.speaker_offset_idx = {}
        prev_speaker = -1
        for f in audios:
            speaker = f.split("/", -1)[-2]
            sig, _ = read_audio(f, trim_silence=self.trim_silence)

            # New speaker, save current chunk and start a fresh chunk
            if prev_speaker == -1 or speaker != prev_speaker:
                self.speaker_offset_idx[self.ids[speaker]] = samples
                print('Speaker {}: start idx: {}'.format(speaker, samples))
                prev_speaker = speaker

            length = sig.size(1)
            # Cut the end of the sample if its too long to be equally split.
            if length % self.sample_length > 0:
                sig = sig[:, :length -(length % self.sample_length)]

            # Split samples
            sigs = sig.view(-1, self.sample_length)
            for sig in sigs:
                sig = sig.unsqueeze(0)
                tensors.append(sig)
                labels.append(self.ids[speaker])
                self.max_len = sig.size(1) if sig.size(1) > self.max_len else self.max_len
                samples += 1
                # Save to chunk-file
                # if len(tensors) == self.chunk_size:
                #     self.save_chunk(chunk_id, lengths, tensors, labels)
                #     chunk_id += 1
                #     tensors = []
                #     labels = []
                    

        # Save all to one chunk-file
        if len(tensors) > 0 :
            self.save_raw_chunk(chunk_id, tensors, labels)
        
        self._write_info(samples)

        if not self.dev_mode:
            shutil.rmtree(raw_abs_dir, ignore_errors=True)
        torchaudio.shutdown_sox()
        print('Done!')

    def save_raw_chunk(self, chunk_id, tensors, labels):
        print('Saving chunk {} with speakers {}'.format(chunk_id, set(labels)))
        data = (tensors, labels)
        torch.save(
            data,
            os.path.join(
                self.root,
                self.processed_folder,
                "vcc2016_raw_train_{:04d}.pt".format(chunk_id)
            )
        )

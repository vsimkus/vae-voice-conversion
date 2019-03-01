import torch.utils.data as data
import numpy as np
import os
import os.path
import shutil
import errno
import torch
import torchaudio
from random import shuffle

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_manifest(dir, shuffle_order=False):
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
    
    if shuffle_order:
        print('Shuffling audio sample order.')
        shuffle(audios)

    return audios


def read_audio(fp, downsample=True):
    if downsample:
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fp)
        E.append_effect_to_chain("gain", ["-h"])
        E.append_effect_to_chain("channels", [1])
        E.append_effect_to_chain("rate", [16000])
        E.append_effect_to_chain("gain", ["-rh"])
        E.append_effect_to_chain("dither", ["-s"])
        sig, sr = E.sox_build_flow_effects()
    else:
        sig, sr = torchaudio.load(fp)
    sig = sig.contiguous()
    return sig, sr


def load_txts(dir):
    """Create a dictionary with all the text of the audio transcriptions."""
    utterences = dict()
    txts = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if fname.endswith(".txt"):
                    with open(os.path.join(root, fname), "r") as f:
                        fname_no_ext = os.path.basename(
                            fname).rsplit(".", 1)[0]
                        utterences[fname_no_ext] = f.readline()
    return utterences


def load_ids(dir):
    """Create a dictionary to convert VCTK participant ID to integer ID."""
    ids = dict()
    id = 0
    dir = os.path.join(os.path.expanduser(dir), 'wav48/')
    for speaker in sorted(os.listdir(dir)):
        d = os.path.join(dir, speaker)
        if not os.path.isdir(d):
            continue

        fname_no_ext = os.path.basename(
            d).rsplit(".", 1)[0]
        ids[fname_no_ext] = id
        id += 1

    return ids


class VCTKPreprocessor():
    """`VCTK Preprocessor for <http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html>`.
    `alternate url <http://datashare.is.ed.ac.uk/handle/10283/2651>`
    Based on torchaudio vctk.py <https://github.com/pytorch/audio>

    Args:
        root (string): Root directory of dataset where the dataset should be stored in vctk/raw/, vctk/processed/ directories.
        shuffle_order (bool, optional): if true, shuffle the audio files across the chunk-files.
        dev_mode(bool, optional): if true, clean up is not performed on raw
            files.  Useful to keep raw audio and transcriptions.
    """
    raw_folder = 'vctk/raw'
    processed_folder = 'vctk/processed'
    zip_path = 'VCTK-Corpus.zip'  # path to local zip file
    dset_path = 'VCTK-Corpus'

    def __init__(self, root, downsample=True, shuffle_order=False, dev_mode=True):
        self.root = os.path.expanduser(root)
        self.downsample = downsample
        self.shuffle_order = shuffle_order
        self.dev_mode = dev_mode
        self.data = []
        self.labels = []
        self.chunk_size = 1000
        self.num_samples = 0
        self.max_len = 0
        self.mean_len = 0.
        self.std_len = 0.

        self.process()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "vctk_info.txt"))

    def _write_info(self, num_items):
        info_path = os.path.join(
            self.root, self.processed_folder, "vctk_info.txt")
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("max_len,{}\n".format(self.max_len))
            f.write("mean_len,{:.4f}\n".format(self.mean_len))
            f.write("std_len,{:.4f}\n".format(self.std_len))
            f.write("ids,{}\n".format(self.ids))

    def process(self):
        """Process the VCTK data if it doesn't exist in processed_folder already."""
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
            os.path.join(dset_abs_path, "COPYING"),
            os.path.join(processed_abs_dir, "VCTK_COPYING")
        )
        audios = make_manifest(dset_abs_path, self.shuffle_order)
        self.ids = load_ids(dset_abs_path)
        self.max_len = 0
        all_lengths = []
        print("Found {} audio files".format(
            len(audios)))
        for n in range(len(audios) // self.chunk_size + 1):
            tensors = []
            labels = []
            lengths = []
            st_idx = n * self.chunk_size
            end_idx = st_idx + self.chunk_size
            for i, f in enumerate(audios[st_idx:end_idx]):
                f_rel_no_ext = os.path.basename(f).rsplit(".", 1)[0]
                sig = read_audio(f, downsample=self.downsample)[0]
                tensors.append(sig)
                lengths.append(sig.size(1))
                labels.append(self.ids[f_rel_no_ext.split('_')[0]])
                self.max_len = sig.size(1) if sig.size(
                    1) > self.max_len else self.max_len
                all_lengths.append(sig.size(1))
            # sort sigs/labels: longest -> shortest
            tensors, labels = zip(*[(b, c) for (a, b, c) in sorted(
                zip(lengths, tensors, labels), key=lambda x: x[0], reverse=True)])
            data = (tensors, labels)
            torch.save(
                data,
                os.path.join(
                    self.root,
                    self.processed_folder,
                    "vctk_{:04d}.pt".format(n)
                )
            )
        self.mean_len = np.mean(all_lengths)
        self.std_len = np.std(all_lengths, ddof=1)
        self._write_info((n * self.chunk_size) + i + 1)
        if not self.dev_mode:
            shutil.rmtree(raw_abs_dir, ignore_errors=True)
        torchaudio.shutdown_sox()
        print('Done!')

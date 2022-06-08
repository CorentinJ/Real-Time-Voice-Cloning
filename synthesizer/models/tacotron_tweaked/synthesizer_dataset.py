from itertools import chain

import librosa
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from synthesizer.utils.text import text_to_sequence
from synthesizer.g2p import g2p_main
from synthesizer.models.tacotron.hparams import hparams
from encoder import inference as encoder
from synthesizer.models.tacotron import audio


class SynthesizerDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, embed_dir: Path, hparams):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, embed_dir))

        with metadata_fpath.open("r", encoding="utf8") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]

        mel_fnames = [x[1] for x in metadata if int(x[4])]
        mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        embed_fnames = [x[2] for x in metadata if int(x[4])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        self.samples_fpaths = list(zip(mel_fpaths, embed_fpaths))
        self.samples_texts = [x[5].strip() for x in metadata if int(x[4])]
        self.metadata = metadata
        self.hparams = hparams
        self.debug = False  # hardcoded but if you can't code you don't need debug..right?

        print("Found %d samples" % len(self.samples_fpaths))

    def __getitem__(self, index):
        mel_path, embed_path = self.samples_fpaths[index]
        try:

            mel = np.load(mel_path, allow_pickle=True).T.astype(np.float32)
        except:
            try:
                mel = np.load(str(mel_path), allow_pickle=True).T.astype(np.float32)
            except:
                mel_path, embed_path = self.samples_fpaths[index+1]
                mel = np.load(mel_path, allow_pickle=True).T.astype(np.float32)
        # Load the embed
        embed = np.load(embed_path)
        # Get the text and (not) clean it
        # с моим заказом возникли проблемы
        # ['s', '<eos>', 'm', 'o0', 'i1', 'm', '<eos>', 'z', 'a0', 'k', 'a1', 'z', 'o0', 'm', '<eos>', 'v', 'o1', 'z',  'nj', 'i0', 'k', 'lj', 'i0', '<eos>', 'p', 'r', 'o0', 'b', 'lj', 'e1', 'm', 'y0', '<eos>']
        text = text_to_sequence(g2p_main(self.samples_texts[index].lower()))[:-1]
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        return text, mel, embed.astype(np.float32), index

    def __len__(self):
        return len(self.samples_fpaths)


class JITSynthesizerDataset(Dataset):
    # Experimental feature
    def __init__(self):
        datasets_name = "LibriTTS"
        datasets_root = Path("datasets\\golos\\train_opus")  # your path here
        subfolders = "train-clean-100"
        dataset_root = datasets_root.joinpath(datasets_name)
        input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
        speaker_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
        print("loading wav_fpaths..")
        self.wav_fpaths = flat([[book_dir.glob("*.opus") for book_dir in x.glob("*")] for x in speaker_dirs])
        self.wav_fpaths = flat([[y for y in x] for x in self.wav_fpaths])
        self.model = encoder.load_model(Path("saved_models/default/encoder.pt")).cpu()
        print("fpaths loaded, length is", len(self.wav_fpaths))

    def __getitem__(self, index):
        wav, _ = librosa.load(str(self.wav_fpaths[index]), sr=hparams.sample_rate)
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max

        text_fpath = self.wav_fpaths[index].with_suffix(".txt")
        with text_fpath.open("r", encoding="utf8") as text_file:
            text = "".join([line for line in text_file])
        text = text.replace("\"", "")
        text = text.strip()
        text = text_to_sequence(g2p_main(text.lower()))[:-1]
        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)
        mel_spectrogram = audio.melspectrogram(wav, hparams).T.T.astype(np.float32)
        embedded_utterance = encoder.embed_utterance(wav, model=self.model).astype(np.float32)
        # print(text, mel_spectrogram.T.astype(np.float32).shape, embedded_utterance.shape, index)
        return text, mel_spectrogram, embedded_utterance, index

    def __len__(self):
        return len(self.wav_fpaths)


def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)


def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)


def collate_synthesizer(batch, r, hparams):
    # Text
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    # Mel spectrogram
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

        # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
    # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
    if hparams.symmetric_mels:
        mel_pad_value = -1 * hparams.max_abs_value
    else:
        mel_pad_value = 0

    mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = np.stack(mel)

    # Speaker embedding (SV2TTS)
    embeds = np.array([x[2] for x in batch])

    # Index (for vocoder preprocessing)
    indices = [x[3] for x in batch]

    # Convert all to tensor
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    embeds = torch.tensor(embeds)

    return chars, mel, embeds, indices


def flat(xss):
    return [x for xs in xss for x in xs]

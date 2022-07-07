from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from synthesizer.utils.text import text_to_sequence


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
                mel_path, embed_path = self.samples_fpaths[index + 1]
                mel = np.load(mel_path, allow_pickle=True).T.astype(np.float32)
        # Load the embed
        embed = np.load(embed_path)

        text = self.samples_texts[index].lower()
        text = text_to_sequence(text)
        text = torch.tensor(text, dtype=torch.int)
        # Convert the list returned by text_to_sequence to a numpy array
        return text, mel, embed.astype(np.float32), index

    def __len__(self):
        return len(self.samples_fpaths)


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

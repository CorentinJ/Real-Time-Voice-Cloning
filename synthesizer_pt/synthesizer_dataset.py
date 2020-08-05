from torch.utils.data import Dataset
from pathlib import Path
from vocoder import audio
from synthesizer_pt import hparams as hp
from synthesizer_pt.utils.text import text_to_sequence
import numpy as np
import torch


class SynthesizerDataset(Dataset):
    def __init__(self, metadata_fpath: Path, mel_dir: Path, embed_dir: Path):
        print("Using inputs from:\n\t%s\n\t%s\n\t%s" % (metadata_fpath, mel_dir, embed_dir))
        
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split("|") for line in metadata_file]
        
        mel_fnames = [x[1] for x in metadata if int(x[4])]
        mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        embed_fnames = [x[2] for x in metadata if int(x[4])]
        embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        self.samples_fpaths = list(zip(mel_fpaths, embed_fpaths))
        self.samples_texts = [x[5].strip() for x in metadata if int(x[4])]
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        # Sometimes index may be a list of 2 (not sure why this happens)
        # If that is the case, return a single item corresponding to first element in index
        if index is list:
            index = index[0]

        mel_path, embed_path = self.samples_fpaths[index]

        # For debugging
        #print(index)
        #print(mel_path)
        #print(embed_path)
        
        # Load the mel spectrogram (range adjusted to [-1, 1] during preprocessing)
        mel = np.load(mel_path).T.astype(np.float32)
        
        # Load the embed
        embed = np.load(embed_path)

        # Get the text and clean it
        text = text_to_sequence(self.samples_texts[index], hp.tts_cleaner_names)
        
        # For debugging
        #from synthesizer_pt.utils.text import symbols
        #print([symbols[i] for i in text])
        #print(text)

        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        return text, mel.astype(np.float32), embed.astype(np.float32)

    def __len__(self):
        return len(self.samples_fpaths)


def collate_synthesizer(batch, r):

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
    # SV2TTS: Pad with -1*max_abs_value instead as that represents silence in our saved mels
    mel = [pad2d(x[1], max_spec_len, pad_value=-1*hp.max_abs_value) for x in batch]
    mel = np.stack(mel)

    # Speaker embedding (SV2TTS)
    embeds = [x[2] for x in batch]


    # Convert all to tensor
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    embeds = torch.tensor(embeds)

    return chars, mel, embeds

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode='constant', constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant', constant_values=pad_value)

from torch.utils.data import Dataset
from pathlib import Path
from vocoder import audio
from synthesizer_pt import hparams as hp
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
        self.samples_texts = [x[5] for x in metadata if int(x[4])] 
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):  
        mel_path, embed_path = self.samples_fpaths[index]
        
        # Load the mel spectrogram and adjust its range to [-1, 1]
        mel = np.load(mel_path).T.astype(np.float32) / hp.mel_max_abs_value
        
        # Load the embed
        embed = np.load(embed_path)
        
        # Get the text
        text = self.samples_texts[index]

        return text, mel.astype(np.float32), embed.astype(np.float32)

    def __len__(self):
        return len(self.samples_fpaths)


def collate_synthesizer(batch, r):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1 
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r 

    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)

    embeds = [x[2] for x in batch]

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)

    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.
    return chars, mel, embeds

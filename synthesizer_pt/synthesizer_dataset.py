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


def collate_synthesizer(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = audio.label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL' :
        y = audio.label_2_float(y.float(), bits)

    return x, y, mels

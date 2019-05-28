from torch.utils.data import Dataset
import numpy as np
from vocoder.params import *
from vocoder import audio
from pathlib import Path


class VocoderDataset(Dataset):
    def __init__(self, metadata_fpath: Path, gta_dir: Path, wav_dir: Path):
        with metadata_fpath.open("r") as metadata_file:
            metadata = [line.split('|') for line in metadata_file]
        
        min_total_n_frames = mel_win + 2 * pad + min_n_frames
        gta_fnames = [x[1] for x in metadata if int(x[4]) >= min_total_n_frames]
        gta_fpaths = [gta_dir.joinpath(fname) for fname in gta_fnames]
        wav_fnames = [x[0] for x in metadata if int(x[4]) >= min_total_n_frames]
        wav_fpaths = [wav_dir.joinpath(fname) for fname in wav_fnames]
        self.samples_fpaths = list(zip(gta_fpaths, wav_fpaths))
        
        print("Found %d samples" % len(self.samples_fpaths))
    
    def __getitem__(self, index):
        mel_path, wav_path = self.samples_fpaths[index]
        
        # Load the wav and quantize it
        wav = np.load(wav_path)
        if use_mu_law:
            wav = audio.compand_signal(wav)
        quant = audio.quantize_signal(wav)
        
        # Load the mel spectrogram and adjust its range to [0, 1] 
        mel = np.load(mel_path).T.astype(np.float32)
        mel = audio.normalize_mel(mel)
        
        return mel, quant
    
    def __len__(self):
        return len(self.samples_fpaths)
    
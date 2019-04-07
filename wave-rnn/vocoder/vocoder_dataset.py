from torch.utils.data import Dataset
from vlibs import fileio
import numpy as np
from vocoder.params import *
from vocoder import audio


class VocoderDataset(Dataset):
    def __init__(self, data_path):
        metadata_fpath = fileio.join(data_path, "synthesized.txt")
        metadata = [line.split('|') for line in fileio.read_all_lines(metadata_fpath)]
        
        min_total_n_frames = mel_win + 2 * pad + min_n_frames
        mel_fnames = [x[1] for x in metadata if int(x[4]) >= min_total_n_frames]
        mel_fpaths = fileio.join(data_path, 'gta', mel_fnames)
        wav_fnames = [x[0] for x in metadata if int(x[4]) >= min_total_n_frames]
        wav_fpaths = fileio.join(data_path, 'audio', wav_fnames)
        self.samples_fpaths = list(zip(mel_fpaths, wav_fpaths))
    
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
    
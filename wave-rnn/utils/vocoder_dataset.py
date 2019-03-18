from torch.utils.data import Dataset
from vlibs import fileio
import numpy as np
from params import *
from utils import audio

class VocoderDataset(Dataset):
    def __init__(self, data_path):
        metadata_fpath = fileio.join(data_path, "vocoder_train.txt")
        metadata = [line.split('|') for line in fileio.read_all_lines(metadata_fpath)]
        
        min_total_n_frames = mel_win + 2 * pad + min_n_frames
        mel_fnames = [x[3] for x in metadata if int(x[4]) >= min_total_n_frames]
        mel_fpaths = fileio.join(data_path, 'gta', mel_fnames)
        wav_fnames = [x[0] for x in metadata if int(x[4]) >= min_total_n_frames]
        wav_fpaths = fileio.join(data_path, 'audio', wav_fnames)
        self.samples_fpaths = list(zip(mel_fpaths, wav_fpaths))
    
    def __getitem__(self, index):
        mel_path, wav_path = self.samples_fpaths[index]
        
        # Load the wav and quantize it
        wav = np.load(wav_path)
        quant = audio.quantize_signal(wav)
        
        # Sanity check
        assert np.min(quant) >= 0 and np.max(quant) < 2 ** bits
        
        # Load the mel spectrogram and adjust its range to [0, 1] 
        mel = np.load(mel_path).T.astype(np.float32)
        mel = mel / (mel_max_abs_value * 2) + 0.5
        mel = np.clip(mel, 0, 1)
        
        return mel, quant
    
    def __len__(self):
        return len(self.samples_fpaths)
    
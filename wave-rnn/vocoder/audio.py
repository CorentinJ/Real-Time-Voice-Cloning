import numpy as np
import librosa
from vocoder.params import *
from scipy.io import wavfile

def load_wav(fpath):
    return librosa.load(fpath, sr=sample_rate)[0]

def save_wav(path, wav):
    wavfile.write(path, sample_rate, (wav * 32767).astype(np.int16))

def normalize_mel(mel):
    """
    Adapts the range of the spectrogram output from the synthesizer to that of the vocoder
    """
    mel = mel / (mel_max_abs_value * 2) + 0.5
    return np.clip(mel, 0, 1)

def compand_signal(wav):
    """
    Applies the mu-law to an audio waveform
    """
    return np.sign(wav) * np.log(1 + (2 ** bits - 1) * np.abs(wav)) / np.log(1 + (2 ** bits - 1))

def quantize_signal(wav):
    """
    Encodes a floating point audio waveform (-1 < wav < 1) to an integer signal (0 <= wav < 2^bits)
    """
    return ((wav + 1.) * (2 ** bits - 1) / 2).astype(np.int)

def restore_signal(wav):
    """
    Decodes an integer signal (0 <= wav < 2^bits) to a floating point audio waveform (-1 < wav < 1)
    """
    return 2 * wav.astype(np.float32) / (2 ** bits - 1.) - 1.

def expand_signal(wav):
    """
    Applies the inverse mu-law to an audio waveform
    """
    return np.sign(wav) * (1 / (2 ** bits - 1)) * ((1 + (2 ** bits - 1)) ** np.abs(wav) - 1)

def split_signal(x):
    unsigned = x + 2 ** 15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2 ** 15

def encode_16bits(x):
    return np.clip(x * 2 ** 15, -2 ** 15, 2 ** 15 - 1).astype(np.int16)

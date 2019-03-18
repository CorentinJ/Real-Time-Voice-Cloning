import numpy as np
import librosa
from params import *

def load_wav(fpath):
    return librosa.load(fpath, sr=sample_rate)[0]

def save_wav(fpath, wav):
    librosa.output.write_wav(fpath, wav, sample_rate)
    
def quantize_signal(wav):
    """
    Encodes a floating point audio waveform (-1 < wav < 1) to an integer signal (0 <= wav < 2^bits)
    """
    if use_mu_law:
        wav = np.sign(wav) * np.log(1 + (2 ** bits - 1) * np.abs(wav)) / np.log(1 + (2 ** bits - 1))
    return np.interp(wav, (-1, 1), (0, 2 ** bits)).astype(np.int)

def restore_signal(wav):
    """
    Decodes an integer signal (0 <= wav < 2^bits) to a floating point audio waveform (-1 < wav < 1)
    """
    wav = np.interp(wav.astype(np.float32), (0, 2 ** bits), (-1, 1))
    if use_mu_law:
        wav = np.sign(wav) * (1 / (2 ** bits - 1)) * ((1 + (2 ** bits - 1)) ** np.abs(wav) - 1) 
    return wav

def split_signal(x):
    unsigned = x + 2 ** 15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2 ** 15

def encode_16bits(x):
    return np.clip(x * 2 ** 15, -2 ** 15, 2 ** 15 - 1).astype(np.int16)

# mel_basis = None
# 
# def linear_to_mel(spectrogram):
#     global mel_basis
#     if mel_basis is None:
#         mel_basis = build_mel_basis()
#     return np.dot(mel_basis, spectrogram)
# 
# def build_mel_basis():
#     return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin)
# 
# def normalize(S):
#     return np.clip((S - min_level_db) / -min_level_db, 0, 1)
# 
# def denormalize(S):
#     return (np.clip(S, 0, 1) * -min_level_db) + min_level_db
# 
# def amp_to_db(x):
#     return 20 * np.log10(np.maximum(1e-5, x))
# 
# def db_to_amp(x):
#     return np.power(10.0, x * 0.05)
# 
# def spectrogram(y):
#     raise Exception()
#     D = stft(y)
#     S = amp_to_db(np.abs(D)) - ref_level_db
#     return normalize(S)
# 
# def melspectrogram(y):
#     raise Exception()
#     D = stft(y)
#     S = amp_to_db(linear_to_mel(np.abs(D)))
#     return normalize(S)
# 
# def stft(y):
#     raise Exception()
#     return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)